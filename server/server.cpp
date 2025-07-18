#include <iostream>
#include <string>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>
#include <vector>
#include <chrono>
#include <thread>
#include <fstream>
#include <filesystem>
#include <random>
#include <sstream>
#include <regex>

// System headers
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>

// httplib
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"

namespace fs = std::filesystem;

// Configuration
const std::string AS_PATH = "aarch64-none-linux-gnu-as";
const std::string LD_PATH = "aarch64-none-linux-gnu-ld";
const int DEFAULT_CPU_CORE = 0; // CPU core to pin execution to

struct CommandResult {
    std::string stdout_output;
    std::string stderr_output;
    int exit_code;
};

// Generate unique temporary filename
std::string generateTempFilename(const std::string& prefix, const std::string& suffix) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(100000, 999999);
    return "/tmp/" + prefix + std::to_string(dis(gen)) + suffix;
}

// Escape JSON string
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
                if (c >= 0 && c < 32) {
                    // Skip control characters
                } else {
                    output += c;
                }
                break;
        }
    }
    return output;
}

// Execute command with timeout and optional CPU pinning
CommandResult executeCommand(const std::string& cmd, int timeout_seconds = 5, int cpu_core = -1) {
    int stdout_pipe[2];
    int stderr_pipe[2];
    pid_t pid;

    if (pipe(stdout_pipe) == -1 || pipe(stderr_pipe) == -1) {
        perror("pipe");
        return {"", "pipe() failed", -1};
    }

    pid = fork();
    if (pid == -1) {
        perror("fork");
        return {"", "fork() failed", -1};
    }

    if (pid == 0) { // Child Process
        close(stdout_pipe[0]);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        close(stdout_pipe[1]);

        close(stderr_pipe[0]);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stderr_pipe[1]);

        // Build command with CPU pinning if requested
        std::string final_cmd = cmd;
        if (cpu_core >= 0) {
            final_cmd = "taskset -c " + std::to_string(cpu_core) + " " + cmd;
        }

        execl("/bin/sh", "sh", "-c", final_cmd.c_str(), (char*) NULL);
        perror("execl");
        _exit(127);
    } else { // Parent Process
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        CommandResult result;
        bool timeout_exceeded = false;
        auto start_time = std::chrono::steady_clock::now();

        while (true) {
            int status;
            pid_t wait_result = waitpid(pid, &status, WNOHANG);

            if (wait_result == pid) {
                if (WIFEXITED(status)) {
                    result.exit_code = WEXITSTATUS(status);
                } else {
                    result.exit_code = -1;
                }
                break;
            }

            if (wait_result == 0) {
                auto current_time = std::chrono::steady_clock::now();
                auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();

                if (elapsed_seconds >= timeout_seconds) {
                    kill(pid, SIGKILL);
                    waitpid(pid, &status, 0);
                    timeout_exceeded = true;
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            } else {
                perror("waitpid");
                result.exit_code = -1;
                result.stderr_output = "waitpid() failed";
                break;
            }
        }

        // Set pipes to non-blocking
        fcntl(stdout_pipe[0], F_SETFL, O_NONBLOCK);
        fcntl(stderr_pipe[0], F_SETFL, O_NONBLOCK);

        // Read output
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
            result.exit_code = 124;
        }

        return result;
    }
}

// Extract CPU cycles from output
long long extractCpuCycles(const std::string& output) {
    // Look for pattern "CPU Cycles: <number>"
    std::regex cycle_pattern(R"(CPU Cycles:\s*(\d+))");
    std::smatch match;
    
    if (std::regex_search(output, match, cycle_pattern)) {
        return std::stoll(match[1].str());
    }
    
    return -1; // Not found
}

// Clean up temporary files
void cleanupFiles(const std::vector<std::string>& files) {
    for (const auto& file : files) {
        try {
            fs::remove(file);
        } catch (...) {
            // Ignore errors during cleanup
        }
    }
}

// Process ARM assembly code
struct ProcessResult {
    std::string stdout_output;
    std::string stderr_output;
    int exit_code;
    long long cpu_cycles;
};

ProcessResult processArmAssembly(const std::string& asm_code, int timeout_seconds = 10) {
    ProcessResult result;
    result.cpu_cycles = -1;
    
    // Generate temporary filenames
    std::string asm_file = generateTempFilename("arm_asm_", ".s");
    std::string obj_file = generateTempFilename("arm_obj_", ".o");
    std::string exe_file = generateTempFilename("arm_exe_", "");
    
    std::vector<std::string> temp_files = {asm_file, obj_file, exe_file};
    
    try {
        // Write assembly code to file
        std::ofstream out(asm_file);
        if (!out) {
            result.stderr_output = "Failed to create assembly file";
            result.exit_code = -1;
            return result;
        }
        out << asm_code;
        out.close();
        
        // Assemble
        std::string as_cmd = AS_PATH + " -o " + obj_file + " " + asm_file;
        CommandResult as_result = executeCommand(as_cmd, timeout_seconds / 3);
        
        if (as_result.exit_code != 0) {
            result.stdout_output = as_result.stdout_output;
            result.stderr_output = "Assembly failed:\n" + as_result.stderr_output;
            result.exit_code = as_result.exit_code;
            cleanupFiles(temp_files);
            return result;
        }
        
        // Link
        std::string ld_cmd = LD_PATH + " -o " + exe_file + " " + obj_file;
        CommandResult ld_result = executeCommand(ld_cmd, timeout_seconds / 3);
        
        if (ld_result.exit_code != 0) {
            result.stdout_output = ld_result.stdout_output;
            result.stderr_output = "Linking failed:\n" + ld_result.stderr_output;
            result.exit_code = ld_result.exit_code;
            cleanupFiles(temp_files);
            return result;
        }
        
        // Execute with CPU pinning
        CommandResult exec_result = executeCommand(exe_file, timeout_seconds / 2, DEFAULT_CPU_CORE);
        
        result.stdout_output = exec_result.stdout_output;
        result.stderr_output = exec_result.stderr_output;
        result.exit_code = exec_result.exit_code;
        
        // Extract CPU cycles from output
        result.cpu_cycles = extractCpuCycles(exec_result.stdout_output);
        
    } catch (const std::exception& e) {
        result.stderr_output = "Exception: " + std::string(e.what());
        result.exit_code = -1;
    }
    
    // Cleanup
    cleanupFiles(temp_files);
    
    return result;
}

int main() {
    httplib::Server svr;

    // Main endpoint for ARM assembly execution
    svr.Post("/", [](const httplib::Request& req, httplib::Response& res) {
        std::string asm_code = req.body;

        if (!asm_code.empty()) {
            std::cout << "Processing ARM assembly code..." << std::endl;

            ProcessResult result = processArmAssembly(asm_code, 10);

            // Build JSON response
            std::string json_response = "{";
            json_response += "\"stdout\": \"" + escapeJsonString(result.stdout_output) + "\",";
            json_response += "\"stderr\": \"" + escapeJsonString(result.stderr_output) + "\",";
            json_response += "\"exit_code\": " + std::to_string(result.exit_code) + ",";
            json_response += "\"cpu_cycles\": " + std::to_string(result.cpu_cycles);
            json_response += "}";

            res.set_content(json_response, "application/json");
        } else {
            res.status = 400;
            res.set_content("{\"error\": \"No assembly code provided in the POST body.\"}", "application/json");
            std::cout << "Invalid request: empty body." << std::endl;
        }
    });

    // Health check endpoint
    svr.Get("/health", [](const httplib::Request& req, httplib::Response& res) {
        res.set_content("{\"status\": \"ok\"}", "application/json");
    });

    std::cout << "ARM Assembly Execution Server listening on port 8080..." << std::endl;
    std::cout << "Using assembler: " << AS_PATH << std::endl;
    std::cout << "Using linker: " << LD_PATH << std::endl;
    std::cout << "CPU pinning to core: " << DEFAULT_CPU_CORE << std::endl;
    
    if (!svr.listen("0.0.0.0", 8080)) {
        std::cerr << "Failed to start server" << std::endl;
        return 1;
    }

    return 0;
}