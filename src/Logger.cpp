#include "Logger.hpp"


void Logger::Init(spdlog::level::level_enum log_level)
{
//    console_->set_pattern();
    console_->set_level(log_level);
}
