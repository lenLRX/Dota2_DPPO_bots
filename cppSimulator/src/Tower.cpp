#include "Tower.h"
#include "simulatorImp.h"
#include <unordered_map>

//TODO use json
static std::unordered_map<std::string, std::unordered_map<std::string, double> > TowerData;

static int init_TowerData = [&]()->int {
    TowerData["Tier1Tower"] = {
        { "HP",14000000 },
        { "MP",0 },
        { "MovementSpeed",0 },
        { "Armor", 14 },
        { "Attack", 120 },
        { "AttackRange",700 },
        { "SightRange", 1900 },
        { "Bounty", 36 },
        { "bountyEXP", 0 },
        { "BaseAttackTime", 1 },
        { "AttackSpeed", 100 }
    };
    return 0;
}();

Tower::Tower(cppSimulatorImp* _Engine,
    Side _side, std::string type_name, pos_tup init_loc)
{
    Engine = _Engine;
    side = _side;
    const auto& data = TowerData[type_name];
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

    viz_radius = 5;
    if (side == Side::Radiant) {
        color = Config::Radiant_Colors;
    }
    else {
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

Tower::~Tower()
{
}

void Tower::step()
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
    }
}

void Tower::draw()
{
}

void Tower::initTowers(cppSimulatorImp* Engine)
{
    Engine->addSprite(new Tower(Engine, Side::Radiant, "Tier1Tower", pos_tup( -1661,-1505 )));
    Engine->addSprite(new Tower(Engine, Side::Radiant, "Tier1Tower", pos_tup( -6254, 1823 )));
    Engine->addSprite(new Tower(Engine, Side::Radiant, "Tier1Tower", pos_tup( 4922, -6122 )));
    Engine->addSprite(new Tower(Engine, Side::Dire, "Tier1Tower", pos_tup( 1032, 359 )));
    Engine->addSprite(new Tower(Engine, Side::Dire, "Tier1Tower", pos_tup( -4706, 6022 )));
    Engine->addSprite(new Tower(Engine, Side::Dire, "Tier1Tower", pos_tup( 6242, -1610 )));
}
