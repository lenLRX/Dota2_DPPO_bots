#include "Creep.h"
#include "simulatorImp.h"

#include <string>

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
        {"bountyEXP", 40},
        {"BaseAttackTime", 1},
        {"AttackSpeed", 100}
    };
    return 0;
}();



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
        PyObject* kwargs = Py_BuildValue("{s:s}", "fill", color);
        v_handle = PyObject_Call(create_rectangle, args, kwargs);
        Py_DECREF(kwargs);
        Py_DECREF(args);
        Py_DECREF(create_rectangle);
        /*
        v_handle = PyObject_CallMethod(canvas,
            "create_rectangle",
            "(dddd)",
            std::get<0>(p) - viz_radius,
            std::get<1>(p) + viz_radius,
            std::get<0>(p) + viz_radius,
            std::get<1>(p) - viz_radius);
        */
    }
}

Creep::~Creep()
{
    Py_XDECREF(v_handle);
}

void Creep::step()
{
    if (isAttacking()) {
        return;
    }
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
    if (canvas != NULL) {
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
