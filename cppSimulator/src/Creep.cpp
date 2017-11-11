#include "Creep.h"
#include "simulatorImp.h"

#include <string>
#include <random>

//TODO use json
static std::unordered_map<std::string,std::unordered_map<std::string,double> > CreepData;

static int init_CreepData = [&]()->int {
    CreepData["MeleeCreep"] = {
        {"HP",550},
        {"MP",0},
        {"MovementSpeed",325},
        {"Armor", 2},
        {"Attack", 21},
        {"AttackRange",100},
        {"SightRange", 750},
        {"Bounty", 36},
        {"bountyEXP", 57},
        {"BaseAttackTime", 1},
        {"AttackSpeed", 100}
    };

    CreepData["RangedCreep"] = {
        { "HP",300 },
        { "MP",0 },
        { "MovementSpeed",325 },
        { "Armor", 0 },
        { "Attack", 23.5 },
        { "AttackRange",500 },
        { "SightRange", 750 },
        { "Bounty", 36 },
        { "bountyEXP", 69 },
        { "BaseAttackTime", 1 },
        { "AttackSpeed", 100 }
    };
    return 0;
}();

static std::default_random_engine rnd_gen;
std::uniform_int_distribution<int> distribution(1, 10);


Creep::Creep(cppSimulatorImp* _Engine, Side _side, std::string type_name)
{
    Engine = _Engine;
    side = _side;
    const auto& data = CreepData[type_name];
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

    //random atk
    Attack += (distribution(rnd_gen) - 5);

    _update_para();

    viz_radius = 2;
    if (side == Side::Radiant) {
        init_loc = pos_tup(-4899, -4397);
        dest = pos_tup(4165, 3681);
        color = Config::Radiant_Colors;
    }
    else {
        init_loc = pos_tup(4165, 3681);
        dest = pos_tup(-4899, -4397);
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
