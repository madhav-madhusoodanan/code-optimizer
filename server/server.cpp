#include <iostream>
#include <string>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>
#include <vector>
#include <chrono> // Required for time measurement
#include <thread> // Required for sleep

// Required for fork, pipe, dup2, exec, read, close
#include <unistd.h>
// Required for waitpid, kill
#include <sys/wait.h>
#include <signal.h> // Required for SIGKILL
// Required for fcntl to set non-blocking I/O
#include <fcntl.h>


// httplib.h is a single-header library.
// You can get it from: https://github.com/yhirose/cpp-httplib
// Just download the httplib.h file and place it in the same directory.
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"

// --- Struct to hold the results of command execution ---
struct CommandResult {
    std::string stdout_output;
    std::string stderr_output;
    int exit_code;
};

// --- Function to escape special characters for JSON ---
// This ensures that the output from the command doesn't break the JSON format.
std::string escapeJsonString(const std::string& input) {
    std::string output;
    output.reserve(input.length());
    for (char c : input) {
        switch (c) {
            case '"':  output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b";  break;
            case '\f': output += "\\f";  break;
            case '\n': output += "\\n";  break;
            case '\r': output += "\\r";  break;
            case '\t': output += "\\t";  break;
            default:
                // Note: You might want to escape other control characters depending on your needs.
                if (c >= 0 && c < 32) {
                    // Example:
                    // char buffer[8];
                    // snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                    // output += buffer;
                } else {
                    output += c;
                }
                break;
        }
    }
    return output;
}


// --- Function to Execute a Command and Get Separate stdout/stderr with a Timeout ---
// This function uses fork() and pipe() to create a child process,
// captures its output, and enforces a timeout.
CommandResult executeCommand(const char* cmd, int timeout_seconds = 5) {
    int stdout_pipe[2];
    int stderr_pipe[2];
    pid_t pid;

    if (pipe(stdout_pipe) == -1 || pipe(stderr_pipe) == -1) {
        perror("pipe");
        return {"", "pipe() failed in server process", -1};
    }

    pid = fork();
    if (pid == -1) {
        perror("fork");
        return {"", "fork() failed in server process", -1};
    }

    if (pid == 0) { // --- Child Process ---
        // Redirect stdout to the write end of the stdout pipe
        close(stdout_pipe[0]); // Close unused read end
        dup2(stdout_pipe[1], STDOUT_FILENO);
        close(stdout_pipe[1]);

        // Redirect stderr to the write end of the stderr pipe
        close(stderr_pipe[0]); // Close unused read end
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stderr_pipe[1]);

        // Execute the command using a shell
        execl("/bin/sh", "sh", "-c", cmd, (char*) NULL);

        // If execl returns, it must have failed.
        perror("execl");
        _exit(127); // Exit child with an error code
    } else { // --- Parent Process ---
        close(stdout_pipe[1]); // Close unused write end
        close(stderr_pipe[1]); // Close unused write end

        CommandResult result;
        bool timeout_exceeded = false;
        auto start_time = std::chrono::steady_clock::now();

        // Loop to wait for the child process to finish or for the timeout to expire
        while (true) {
            int status;
            // WNOHANG makes waitpid return immediately if the child hasn't exited.
            pid_t wait_result = waitpid(pid, &status, WNOHANG);

            if (wait_result == pid) { // Child has terminated
                if (WIFEXITED(status)) {
                    result.exit_code = WEXITSTATUS(status);
                } else {
                    result.exit_code = -1; // Indicate failure or abnormal termination
                }
                break; // Exit the loop
            }

            if (wait_result == 0) { // Child is still running
                auto current_time = std::chrono::steady_clock::now();
                auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();

                if (elapsed_seconds >= timeout_seconds) {
                    // Timeout exceeded, kill the child process
                    kill(pid, SIGKILL);
                    waitpid(pid, &status, 0); // Clean up the zombie process
                    timeout_exceeded = true;
                    break; // Exit the loop
                }

                // Sleep briefly to avoid busy-waiting and consuming 100% CPU
                std::this_thread::sleep_for(std::chrono::milliseconds(50));

            } else { // waitpid failed
                perror("waitpid");
                result.exit_code = -1;
                result.stderr_output = "waitpid() failed in parent process";
                break;
            }
        }

        // After the loop, drain any remaining output from the pipes.
        // Set pipes to non-blocking to avoid getting stuck here.
        fcntl(stdout_pipe[0], F_SETFL, O_NONBLOCK);
        fcntl(stderr_pipe[0], F_SETFL, O_NONBLOCK);

        std::array<char, 256> buffer;
        ssize_t count;
        while ((count = read(stdout_pipe[0], buffer.data(), buffer.size())) > 0) {
            result.stdout_output.append(buffer.data(), count);
        }
        while ((count = read(stderr_pipe[0], buffer.data(), buffer.size())) > 0) {
            result.stderr_output.append(buffer.data(), count);
        }

        close(stdout_pipe[0]);
        close(stderr_pipe[0]);

        if (timeout_exceeded) {
            result.stderr_output += "\nError: Command execution timed out after " + std::to_string(timeout_seconds) + " seconds.";
            result.exit_code = 124; // Standard exit code for timeout
        }

        return result;
    }
}

// --- Main Server Function ---
int main() {
    // Create an instance of the httplib::Server
    httplib::Server svr;

    // --- Define the handler for POST requests to the root "/" ---
    svr.Post("/", [](const httplib::Request& req, httplib::Response& res) {
        std::string command_to_execute = req.body;

        if (!command_to_execute.empty()) {
            std::cout << "Executing command: " << command_to_execute << std::endl;

            // Execute with a 5-second timeout
            CommandResult result = executeCommand(command_to_execute.c_str(), 5);

            // Manually construct a JSON response string.
            // For more complex JSON, a dedicated library like nlohmann/json is recommended.
            std::string json_response = "{";
            json_response += "\"stdout\": \"" + escapeJsonString(result.stdout_output) + "\",";
            json_response += "\"stderr\": \"" + escapeJsonString(result.stderr_output) + "\",";
            json_response += "\"exit_code\": " + std::to_string(result.exit_code);
            json_response += "}";

            res.set_content(json_response, "application/json");
        } else {
            res.status = 400; // Bad Request
            res.set_content("{\"error\": \"No command provided in the POST body.\"}", "application/json");
            std::cout << "Invalid request received: empty body." << std::endl;
        }
    });

    // --- Start the server ---
    std::cout << "Server listening on port 8080..." << std::endl;
    if (!svr.listen("0.0.0.0", 8080)) {
        std::cerr << "Failed to start server" << std::endl;
        return 1;
    }

    return 0;
}