#include "Creep.h"
#include "simulatorImp.h"

#include <string>
#include <random>

//TODO use json

static SpriteDataType CreepData {
    { "MeleeCreep", {
        {"HP",new double(550)},
        {"MP",new double(0)},
        {"MovementSpeed",new double(325)},
        {"Armor", new double(2)},
        {"Attack", new double(21)},
        {"AttackRange",new double(100)},
        {"SightRange", new double(750)},
        {"Bounty", new double(36)},
        {"bountyEXP", new double(57)},
        {"BaseAttackTime", new double(1)},
        {"AtkPoint", new double(0.467)},
        {"AtkBackswing", new double(0.533)},
        {"AttackSpeed", new double(100)},
        {"ProjectileSpeed", new double(-1) },
        {"atktype", new AtkType(melee)}
    }},

    {"RangedCreep", {
        { "HP",new double(300) },
        { "MP",new double(0) },
        { "MovementSpeed",new double(325) },
        { "Armor", new double(0) },
        { "Attack", new double(23.5) },
        { "AttackRange",new double(500) },
        { "SightRange", new double(750) },
        { "Bounty", new double(36) },
        { "bountyEXP", new double(69) },
        { "BaseAttackTime", new double(1) },
        { "AttackSpeed", new double(100) },
        { "AtkPoint", new double(0.5) },
        { "AtkBackswing", new double(0.67) },
        { "ProjectileSpeed", new double(900) },
        { "atktype", new AtkType(ranged) }
    }}
};

static std::default_random_engine rnd_gen;
static std::uniform_int_distribution<int> distribution(1, 10);
static std::uniform_int_distribution<int> pos_distribution(1, 500);
static std::uniform_int_distribution<int> sign_distribution(-1, 1);

static int get_rand()
{
    return sign_distribution(rnd_gen) * pos_distribution(rnd_gen);
}


Creep::Creep(cppSimulatorImp* _Engine, Side _side, std::string type_name)
{
    Engine = _Engine;
    side = _side;
    const auto& data = CreepData[type_name];
    INIT_ATTR_BY(data);

    //random atk
    Attack += (distribution(rnd_gen) - 5);

    _update_para();

    viz_radius = 2;
    if (side == Side::Radiant) {
        init_loc = pos_tup(-4899 + get_rand(), -4397 + get_rand());
        dest = pos_tup(4165 + get_rand(), 3681 + get_rand());
        color = Config::Radiant_Colors;
    }
    else {
        init_loc = pos_tup(4165 + get_rand(), 3681 + get_rand());
        dest = pos_tup(-4899 + get_rand(), -4397 + get_rand());
        color = Config::Dire_Colors;
    }

    location = init_loc;

    if (Engine->get_canvas() != NULL) {
        canvas = Engine->get_canvas();
        pos_tup p = pos_in_wnd();
        PyObject* create_rectangle = PyObject_GetAttrString(canvas, "create_rectangle");
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

Creep::~Creep()
{
}

void Creep::step()
{
    if (isDead())
        return;
    if (isAttacking())
        return;
    auto nearby_enemy = Engine->get_nearby_enemy(this);
    if (!nearby_enemy.empty()) {
        Sprite* target = nearby_enemy.front().first;
        if (nearby_enemy.front().second < AttackRange) {
            attack(target);
        }
        else {
            set_move(target->get_location());
        }
    }
    else {
        set_move(dest);
    }
}

void Creep::draw()
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
