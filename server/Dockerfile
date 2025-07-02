# --- Stage 1: Builder ---
# We use a specific version of Alpine for reproducibility.
# This stage will contain the build tools needed to compile the C++ application.
# The official Alpine image is multi-architecture and will work for ARM.
FROM alpine:3.22 AS builder

# Set a working directory for the application.
WORKDIR /app

# Install build dependencies: C++ compiler, make, and OpenSSL development libraries.
# --no-cache cleans up the apk cache in the same layer, reducing image size.
RUN apk update && apk add --no-cache \
    g++ \
    make \
    openssl-dev

# It's good practice to download dependencies into the container rather than
# copying them from the host, to make the build more self-contained.
# We download the single-header httplib library.
RUN wget https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h -O ./httplib.h

# Copy the server source code into the builder stage.
COPY server.cpp .

# Compile the application.
# We link against pthread and the OpenSSL libraries (ssl, crypto).
# The output is a dynamically linked executable named 'server'.
RUN g++ -o server server.cpp -pthread -lssl -lcrypto


# --- Stage 2: Final Image ---
# This is the final, hardened image that will run the application.
# It starts from a clean Alpine base to ensure it's minimal.
FROM alpine:3.22

# Install only the required RUNTIME dependencies.
# For our dynamically linked binary, we need the C++ standard library and OpenSSL runtime libraries.
RUN apk update && apk add --no-cache \
    libstdc++ \
    openssl

# Create a dedicated, non-root user and group to run the application.
# This is a critical security measure to limit the container's privileges.
# -S creates a system user (no password, no home directory).
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

# Set the working directory for the final image.
WORKDIR /app

# Copy ONLY the compiled binary from the builder stage.
# This is the core of the multi-stage build pattern. No source code or build tools
# are included in the final image.
COPY --from=builder /app/server .

# Set the ownership of the application directory and binary to the non-root user.
RUN chown -R appuser:appgroup /app

# Switch the active user for all subsequent commands.
USER appuser

# Expose the port the server will listen on.
# This informs Docker that the container listens on this port.
EXPOSE 8080

# --- SECURITY: HOW TO RUN THIS CONTAINER ---
# To protect the host system from fork bombs (a malicious command that rapidly
# replicates itself to exhaust system resources), it is CRITICAL to run this
# container with process limits. This cannot be set at build time in the Dockerfile.
#
# Use the `--ulimit` flag with `docker run` to restrict the number of processes.
#
# Example command:
# docker run -d -p 8080:8080 --ulimit nproc=64:128 --name my-cpp-server cpp-command-server
#
# This sets a soft limit of 64 processes and a hard limit of 128 for the container.

# Define the command to run when the container starts.
CMD ["./server"]
