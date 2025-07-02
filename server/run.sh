# docker buildx build --platform linux/arm64 -t cpp-command-server .
docker service create --replicas 5 --name test-server --ulimit nproc=64:128 -p 8080:8080 cpp-command-server:latest