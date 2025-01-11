#include "Logger.hpp"

#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_sinks.h"


std::shared_ptr<spdlog::logger> CreateLogger() {
    static constexpr auto logger_name = "Logger";

    std::vector<spdlog::sink_ptr> sinks{};

    sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    sinks.push_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>("logfile.txt", 1048576 * 5, 3, false));

    auto spd_logger = std::make_shared<spdlog::logger>(logger_name, begin(sinks), end(sinks));

    spdlog::register_logger(spd_logger);
    spdlog::set_default_logger(spd_logger);

    return spd_logger;
}


void Logger::Init(spdlog::level::level_enum log_level)
{
    console_->set_pattern("%^[%d-%m-%C %R] %l: %v%$");
    console_->set_level(log_level);
}


void GlfwErrorCallback(int error, const char* description)
{
    LOG_ERROR("GLFW Error {}: {}", error, description);
}
