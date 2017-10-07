#include "Hero.h"

#include "simulatorImp.h"
#include <unordered_map>

//TODO use json
static std::unordered_map<std::string, std::unordered_map<std::string, double> > HeroData;

static int init_HeroData = [&]()->int {
    HeroData["ShadowFiend"] = {
        { "HP",500 },
        { "MP",273 },
        { "MovementSpeed",315 },
        { "Armor", 0.86 },
        { "Attack", 21 },
        { "AttackRange",500 },
        { "SightRange", 1800 },
        { "Bounty", 200 },
        { "bountyEXP", 200 },
        { "BaseAttackTime", 1.7 },
        { "AttackSpeed", 120 }
    };
    return 0;
}();

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
        init_loc = pos_tup(-7205, -6610);
        color = Config::Radiant_Colors;
    }
    else {
        init_loc = pos_tup(7000, 6475);
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
        PyObject* kwargs = Py_BuildValue("{s:s}", "fill", color);
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
    move_order = order;
}

PyObject* Hero::get_state_tup()
{
    PyObject* self_input = Py_BuildValue("[dd]",
        std::get<0>(location) / Config::map_div,
        std::get<1>(location) / Config::map_div);
    if (PyErr_Occurred())
        printf("%d\n", __LINE__);
    auto nearby_ally = Engine->get_nearby_ally(this);
    Py_ssize_t ally_input_size = static_cast<Py_ssize_t>(nearby_ally.size());
    PyObject* ally_input = NULL;
    if (ally_input_size > 0) {
        ally_input = PyList_New(ally_input_size);
        for (Py_ssize_t i = 0; i < ally_input_size; ++i) {
            PyList_SetItem(ally_input, i, Py_BuildValue("[dd]",
                std::get<0>(nearby_ally[i].first->get_location()) / Config::map_div,
                std::get<1>(nearby_ally[i].first->get_location()) / Config::map_div));
        }
    }
    else {
        ally_input = Py_BuildValue("[[dd]]", 0.0, 0.0);
    }
    if (PyErr_Occurred())
        printf("%d\n", __LINE__);
    
    PyObject* state = Py_BuildValue("{s:O,s:O}", "self_input", self_input, "ally_input", ally_input);
    if (PyErr_Occurred())
        printf("%d\n", __LINE__);
    double reward = (exp - last_exp) + (HP - last_HP) * 0.1;

    last_exp = exp;
    last_HP = HP;

    PyObject* ret = Py_BuildValue("(OdO)", state, reward, _isDead ? Py_True : Py_False);
    if (PyErr_Occurred())
        printf("%d\n", __LINE__);
    return ret;
}
