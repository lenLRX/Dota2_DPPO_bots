class Config(object):
    delta_time = 1.0
    tick_per_second = 10
    map_div = 7000.0
    rad_init_pos = [-0.95714285714286 * map_div,
            -0.95714341517857 * map_div]
    dire_init_pos = [0.98571428571429 * map_div,
            0.949999441964297 * map_div]
    velocity = 315
    bound_length = 8000.0

    windows_size = 400.0

    game2window_scale = windows_size / bound_length

    Colors = {"Radiant":"green","Dire":"red"}