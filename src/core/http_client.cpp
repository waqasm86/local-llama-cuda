#include "llcuda/http_client.hpp"
#include <string>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <sstream>

namespace llcuda {

HttpClient::Response HttpClient::post(const std::string& url, const std::string& json_body) {
        Response resp;

        // Parse URL (simple http://host:port/path parser)
        std::string host, path;
        int port = 80;

        size_t proto_end = url.find("://");
        size_t start = (proto_end != std::string::npos) ? proto_end + 3 : 0;

        size_t path_start = url.find('/', start);
        if (path_start == std::string::npos) {
            path = "/";
            path_start = url.size();
        } else {
            path = url.substr(path_start);
        }

        std::string host_port = url.substr(start, path_start - start);
        size_t colon_pos = host_port.find(':');
        if (colon_pos != std::string::npos) {
            host = host_port.substr(0, colon_pos);
            port = std::stoi(host_port.substr(colon_pos + 1));
        } else {
            host = host_port;
        }

        // DNS resolution
        struct addrinfo hints{}, *result = nullptr;
        std::memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;

        int status = getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &result);
        if (status != 0 || !result) {
            throw std::runtime_error("DNS resolution failed for " + host);
        }

        // Create socket
        int sockfd = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
        if (sockfd < 0) {
            freeaddrinfo(result);
            throw std::runtime_error("Failed to create socket");
        }

        // Set timeouts
        struct timeval tv{};
        tv.tv_sec = 600;  // 10 minutes for LLM inference
        tv.tv_usec = 0;
        setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

        // Connect
        if (connect(sockfd, result->ai_addr, result->ai_addrlen) < 0) {
            close(sockfd);
            freeaddrinfo(result);
            throw std::runtime_error("Failed to connect to " + host + ":" + std::to_string(port));
        }

        freeaddrinfo(result);

        // Build HTTP request
        std::ostringstream request;
        request << "POST " << path << " HTTP/1.1\r\n";
        request << "Host: " << host << "\r\n";
        request << "Content-Type: application/json\r\n";
        request << "Content-Length: " << json_body.size() << "\r\n";
        request << "Connection: close\r\n";
        request << "\r\n";
        request << json_body;

        std::string req_str = request.str();

        // Send request
        ssize_t sent = send(sockfd, req_str.data(), req_str.size(), 0);
        if (sent < 0 || static_cast<size_t>(sent) != req_str.size()) {
            close(sockfd);
            throw std::runtime_error("Failed to send HTTP request");
        }

        // Receive response
        std::vector<uint8_t> buffer(65536);
        std::vector<uint8_t> response_data;

        while (true) {
            ssize_t n = recv(sockfd, buffer.data(), buffer.size(), 0);
            if (n <= 0) break;
            response_data.insert(response_data.end(), buffer.begin(), buffer.begin() + n);
        }

        close(sockfd);

        if (response_data.empty()) {
            throw std::runtime_error("Empty HTTP response");
        }

        // Parse response
        std::string response_str(response_data.begin(), response_data.end());

        // Extract status code
        size_t status_pos = response_str.find("HTTP/");
        if (status_pos != std::string::npos) {
            size_t code_start = response_str.find(' ', status_pos) + 1;
            size_t code_end = response_str.find(' ', code_start);
            if (code_start != std::string::npos && code_end != std::string::npos) {
                std::string status_code = response_str.substr(code_start, code_end - code_start);
                resp.status = std::stol(status_code);
            }
        }

        // Extract body (after \r\n\r\n)
        size_t body_start = response_str.find("\r\n\r\n");
        if (body_start != std::string::npos) {
            body_start += 4;
            resp.body.assign(response_data.begin() + body_start, response_data.end());
        }

        return resp;
}

} // namespace llcuda
