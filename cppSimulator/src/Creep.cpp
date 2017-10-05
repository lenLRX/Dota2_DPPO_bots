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



Creep::Creep(cppSimulatorImp* Engine, Side side, std::string type_name)
{
    Engine = Engine;
    side = side;
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
    if (Engine->get_canvas() != NULL) {
        pos_tup p = pos_in_wnd();
        v_handle = PyObject_CallMethod(Engine->get_canvas(),
            "create_rectangle",
            "dddd",
            std::get<0>(p) - viz_radius,
            std::get<1>(p) + viz_radius,
            std::get<0>(p) + viz_radius,
            std::get<0>(p) - viz_radius,
            NULL);
        Py_INCREF(v_handle);
    }
}

Creep::~Creep()
{
    Py_XDECREF(v_handle);
}

void Creep::step()
{
    printf("%p stepping\n", this);
}
