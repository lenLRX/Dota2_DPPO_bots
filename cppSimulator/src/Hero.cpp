#include "Hero.h"
#include "Creep.h"
#include "simulatorImp.h"
#include <unordered_map>
#include <cmath>
#include <random>

//TODO use json
static std::unordered_map<std::string, std::unordered_map<std::string, double> > HeroData;

static std::default_random_engine rnd_gen;
static std::uniform_int_distribution<int> pos_distribution(1, 1000);
static std::uniform_int_distribution<int> sign_distribution(-1, 1);

static int init_HeroData = [&]()->int {
    HeroData["ShadowFiend"] = {
        { "HP",200 },
        { "MP",273 },
        { "MovementSpeed",315 },
        { "Armor", 0.86 },
        { "Attack", 21 },
        { "AttackRange",500 },
        { "SightRange", 1800 },
        { "Bounty", 200 },
        { "bountyEXP", 0 },
        { "BaseAttackTime", 1.7 },
        { "AttackSpeed", 120 }
    };
    return 0;
}();

static int get_rand()
{
    return sign_distribution(rnd_gen) * pos_distribution(rnd_gen);
}

Hero::Hero(cppSimulatorImp* _Engine, Side _side, std::string type_name)
{
    Engine = _Engine;
    side = _side;
    const auto& data = HeroData[type_name];
    SETATTR(data, HP);
    SETATTR(data, MP);
    SETATTR(data, MovementSpeed);
    SETATTR(data, BaseAttackTime);
    SETATTR(data, AttackSpeed);
    SETATTR(data, Armor);
    SETATTR(data, Attack);
    SETATTR(data, AttackRange);
    SETATTR(data, SightRange);
    SETATTR(data, Bounty);
    SETATTR(data, bountyEXP);

    last_exp = 0.0;
    last_HP = HP;

    _update_para();

    viz_radius = 5;
    if (side == Side::Radiant) {
        init_loc = pos_tup(-7205 + get_rand(), -6610 + get_rand());
        color = Config::Radiant_Colors;
    }
    else {
        init_loc = pos_tup(7000 + get_rand(), 6475 + get_rand());
        color = Config::Dire_Colors;
    }

    location = init_loc;
    move_order = pos_tup(0,0);

    if (Engine->get_canvas() != NULL) {
        canvas = Engine->get_canvas();
        pos_tup p = pos_in_wnd();
        PyObject* create_rectangle = PyObject_GetAttrString(canvas, "create_oval");
        PyObject* args = Py_BuildValue("(dddd)",
            std::get<0>(p) - viz_radius,
            std::get<1>(p) + viz_radius,
            std::get<0>(p) + viz_radius,
            std::get<1>(p) - viz_radius);
        PyObject* kwargs = Py_BuildValue("{s:s}", "fill", color.c_str());
        v_handle = PyObject_Call(create_rectangle, args, kwargs);
        Py_DECREF(kwargs);
        Py_DECREF(args);
        Py_DECREF(create_rectangle);
    }
}

Hero::~Hero()
{
}

void Hero::step()
{
    auto p = pos_tup(std::get<0>(move_order) + std::get<0>(location),
        std::get<1>(move_order) + std::get<1>(location));
    set_move(p);
}

void Hero::draw()
{
    if (v_handle != NULL) {
        auto p = pos_in_wnd();
        Py_XDECREF(PyObject_CallMethod(canvas,
            "coords",
            "(Odddd)",
            v_handle,
            std::get<0>(p) - viz_radius,
            std::get<1>(p) + viz_radius,
            std::get<0>(p) + viz_radius,
            std::get<1>(p) - viz_radius));
    }
    
}

void Hero::set_move_order(pos_tup order)
{
    int sign = side == Side::Radiant ? 1 : -1;
    move_order = pos_tup(sign * std::get<0>(order),
        sign * std::get<1>(order));
}

static auto is_creep = [](Sprite* s) -> bool { return dynamic_cast<Creep*>(s) != nullptr; };

PyObject* Hero::get_state_tup()
{
    int sign = side == Side::Radiant ? 1 : -1 ;
    
    auto nearby_ally = Engine->get_nearby_ally(this, is_creep);
    size_t ally_input_size = nearby_ally.size();
    double ally_x = 0.0;
    double ally_y = 0.0;
    for (size_t i = 0; i < ally_input_size; ++i) {
        ally_x += sign * (std::get<0>(nearby_ally[i].first->get_location()) - std::get<0>(location)) / Config::map_div;
        ally_y += sign * (std::get<1>(nearby_ally[i].first->get_location()) - std::get<1>(location)) / Config::map_div;
    }

    if (0 != ally_input_size) {
        ally_x /= (double)ally_input_size;
        ally_y /= (double)ally_input_size;
    }

    auto nearby_enemy = Engine->get_nearby_enemy(this, is_creep);
    size_t enemy_input_size = nearby_enemy.size();
    double enemy_x = 0.0;
    double enemy_y = 0.0;
    for (size_t i = 0; i < enemy_input_size; ++i) {
        enemy_x += sign * (std::get<0>(nearby_enemy[i].first->get_location()) - std::get<0>(location)) / Config::map_div;
        enemy_y += sign * (std::get<1>(nearby_enemy[i].first->get_location()) - std::get<1>(location)) / Config::map_div;
    }

    if (0 != enemy_input_size) {
        enemy_x /= (double)enemy_input_size;
        enemy_y /= (double)enemy_input_size;
    }

    PyObject* state = Py_BuildValue("[ddidddddd]",
        sign * std::get<0>(location) / Config::map_div,
        sign * std::get<1>(location) / Config::map_div,
        side,
        ally_x,
        ally_y,
        (double)ally_input_size,
        enemy_x,
        enemy_y,
        (double)enemy_input_size
    );

    if (NULL == state) {
        printf("self_input error!\n");
        return NULL;
    }

    double reward = (exp - last_exp) * 0.01 + (HP - last_HP) * 0.01;

    last_exp = exp;
    last_HP = HP;

    PyObject* ret = Py_BuildValue("(OdO)", state, reward, _isDead ? Py_True : Py_False);

    Py_DECREF(state);

    return ret;
}

PyObject* Hero::predefined_step()
{
    int sign = side == Side::Radiant ? 1 : -1;
    auto nearby_ally = Engine->get_nearby_enemy(this, is_creep);
    pos_tup ret;
    int _dis = 700;
    if (nearby_ally.size() > 0)
    {
        ret = nearby_ally[0].first->get_location();
        if (side == Side::Radiant) {
            ret = pos_tup(std::get<0>(nearby_ally[0].first->get_location()) - _dis,
                std::get<1>(nearby_ally[0].first->get_location()) - _dis);
        }
        else {
            ret = pos_tup(std::get<0>(nearby_ally[0].first->get_location()) + _dis,
                std::get<1>(nearby_ally[0].first->get_location()) + _dis);
        }
    }
    else {
        ret = pos_tup(-482, -400);
    }
    
    double dx = std::get<0>(ret) - std::get<0>(location);
    double dy = std::get<1>(ret) - std::get<1>(location);
    dx *= sign;
    dy *= sign;

    double a = std::atan2(dy, dx);
    PyObject* obj = Py_BuildValue("[dd]", std::cos(a), std::sin(a));
    return obj;
}