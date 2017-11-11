#include "Config.h"

const double Config::tick_per_second = 10.0;
const double Config::map_div =700.0;
const pos_tup Config::rad_init_pos
    = pos_tup(
        -0.95714285714286 * Config::map_div,
        -0.95714341517857 * Config::map_div);
const pos_tup Config::dire_init_pos
    = pos_tup(
        0.98571428571429 * Config::map_div,
        0.949999441964297 * Config::map_div);
const double Config::velocity = 315.0;
const double Config::bound_length = 8000.0;
const double Config::windows_size = 400.0;
const double Config::game2window_scale = Config::windows_size / Config::bound_length;
const char* Config::Radiant_Colors = "green";
const char* Config::Dire_Colors = "red";