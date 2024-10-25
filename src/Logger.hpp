#ifndef HYPERCPP_LOGGER_HPP
#define HYPERCPP_LOGGER_HPP

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <memory>


#ifdef DISABLE_LOGGING
    #define LOG_TRACE(...)
    #define LOG_INFO(...)
    #define LOG_WARN(...)
    #define LOG_ERROR(...)
    #define LOG_CRITICAL(...)
#else
    #define LOG_TRACE(...)    Logger::console_->trace(__VA_ARGS__)
    #define LOG_INFO(...)     Logger::console_->info(__VA_ARGS__)
    #define LOG_WARN(...)     Logger::console_->warn(__VA_ARGS__)
    #define LOG_ERROR(...)    Logger::console_->error(__VA_ARGS__)
    #define LOG_CRITICAL(...) Logger::console_->critical(__VA_ARGS__)
#endif


void GlfwErrorCallback(int error, const char* description);

class Logger
{
public:
    static void Init(spdlog::level::level_enum log_level);

    inline static std::shared_ptr<spdlog::logger> console_ = spdlog::stdout_color_mt("console");
};

#endif //HYPERCPP_LOGGER_HPP
