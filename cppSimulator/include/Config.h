#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <tuple>

enum class Side {
    Radiant = 0,
    Dire = 1
};

typedef std::tuple<double, double> pos_tup;

class Config {
public:
    static const double tick_per_second;
    static const double map_div;
    static const pos_tup rad_init_pos;
    static const pos_tup dire_init_pos;
    static const double velocity;
    static const double bound_length;
    static const double windows_size;
    static const double game2window_scale;
    static const char* Radiant_Colors;
    static const char* Dire_Colors;
};

#endif//__CONFIG_H__