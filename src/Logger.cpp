#include "Logger.hpp"


void Logger::Init(spdlog::level::level_enum log_level)
{
    console_->set_pattern("%^[%d-%m-%C %R] %l: %v%$");
    console_->set_level(log_level);
}


void GlfwErrorCallback(int error, const char* description)
{
    LOG_ERROR("GLFW Error {}: {}", error, description);
}
