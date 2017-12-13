#include "Hero.h"
#include "Creep.h"
#include "log.h"
#include "simulatorImp.h"
#include <cmath>
#include <random>

//TODO use json

static std::default_random_engine rnd_gen;
static std::uniform_int_distribution<int> pos_distribution(1, 1000);
static std::uniform_int_distribution<int> sign_distribution(-1, 1);

static SpriteDataType HeroData {
    { "ShadowFiend", {
        { "HP",new double(500) },
        { "MP",new double(273) },
        { "MovementSpeed",new double(313) },
        { "Armor", new double(0.86) },
        { "Attack", new double(38) },
        { "AttackRange",new double(500) },
        { "SightRange", new double(1800) },
        { "Bounty", new double(200) },
        { "bountyEXP", new double(0) },
        { "BaseAttackTime", new double(1.7) },
        { "AttackSpeed", new double(120) },
        { "AtkPoint", new double(0.5) },
        { "AtkBackswing", new double(0.54) },
        { "ProjectileSpeed", new double(1200) },
        { "atktype", new AtkType(ranged) }
    }}
};

static int get_rand()
{
    return sign_distribution(rnd_gen) * pos_distribution(rnd_gen);
}

Hero::Hero(cppSimulatorImp* _Engine, Side _side, std::string type_name):target(nullptr)
{
    Engine = _Engine;
    side = _side;
    const auto& data = HeroData[type_name];
    INIT_ATTR_BY(data);

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
    LOG << "gold:" << gold << endl;
    Logger::getInstance().flush();
}

void Hero::step()
{
    if (isAttacking())
        return;
    if (decisonType::noop == decision) {
        ;
    }
    else if (decisonType::move == decision) {
        auto p = pos_tup(std::get<0>(move_order) + std::get<0>(location),
            std::get<1>(move_order) + std::get<1>(location));
        set_move(p);
    }
    else if (decisonType::attack == decision) {
        if (nullptr == target) {
            LOG << "null target!\n";
            Logger::getInstance().flush();
            exit(1);
        }
        attack(target);
    }
    
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

void Hero::set_order(PyObject* order)
{
    PyObject* subdecision;
    if (!PyArg_ParseTuple(order, "iO", &decision, &subdecision)) {
        LOG << "Parse Arg error";
        return;
    }
    if (decisonType::noop == decision) {
        ;
    }
    else if (decisonType::move == decision) {
        int sign = side == Side::Radiant ? 1 : -1;
        double x, y;
        if (!PyArg_ParseTuple(subdecision, "dd", &x, &y)) {
            LOG << "Parse Arg error";
            return;
        }
        move_order = pos_tup(sign * x * 1000,
            sign * y * 1000);
    }
    else if (decisonType::attack == decision) {
        target = nullptr;
        int target_idx = PyLong_AsLong(subdecision);
        if (target_idx >= (int)target_list.size()) {
            LOG << "index out of range! target_list size:" << target_list.size() << "," << target_idx << endl;
            exit(4);
        }
        target = target_list[target_idx];
    }
    
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

    PyObject* env_state = Py_BuildValue("[dddidddddd]",
        sign * std::get<0>(location) / Config::map_div,
        sign * std::get<1>(location) / Config::map_div,
        Attack,
        side,
        ally_x,
        ally_y,
        (double)ally_input_size,
        enemy_x,
        enemy_y,
        (double)enemy_input_size
    );

    if (NULL == env_state) {
        printf("env_state error!\n");
        return NULL;
    }

    PyObject* state_targets_list;
    if (enemy_input_size > 0) {
        target_list.clear();
        state_targets_list = PyList_New(enemy_input_size);
        for (int i = 0; i < enemy_input_size; i++) {
            PyList_SET_ITEM(state_targets_list,i, Py_BuildValue("(dd)", nearby_enemy[i].first->get_HP(), nearby_enemy[i].second / AttackRange));
            target_list.push_back(nearby_enemy[i].first);
        }
        //LOG << "target_list.size " << target_list.size() << endl;
    }
    else {
        state_targets_list = Py_BuildValue("[]");
    }

    PyObject* state = Py_BuildValue("{s:O,s:O}", "env_input", env_state, "target_input", state_targets_list);
    Py_XDECREF(env_state);
    Py_XDECREF(state_targets_list);

    double reward = (exp - last_exp) * 0.01 + (HP - last_HP) * 0.01 + (gold - last_gold) * 0.1;

    last_exp = exp;
    last_HP = HP;
    last_gold = gold;

    PyObject* ret = Py_BuildValue("(OdO)", state, reward, _isDead ? Py_True : Py_False);

    Py_DECREF(state);

    return ret;
}

PyObject* Hero::predefined_step()
{
    /*
    //this is real bot should do, but it is hard to train
    if (isAttacking()) {
        Py_INCREF(Py_None);
        PyObject* obj = Py_BuildValue("(iO)", decisonType::noop, Py_None);
        return obj;
    }
    */
    int sign = side == Side::Radiant ? 1 : -1;
    auto nearby_enemy = Engine->get_nearby_enemy(this, is_creep);
    auto nearby_enemy_size = nearby_enemy.size();
    auto targetlist_size = target_list.size();
    if (targetlist_size > 0)
    {
        for (int i = 0; i < targetlist_size; ++i) {
            if (!target_list[i]->isDead() && target_list[i]->get_HP() < Attack) {
                PyObject* obj = Py_BuildValue("(ii)", decisonType::attack, i);
                return obj;
            }
        }
    }
    pos_tup ret;
    int _dis = 700;
    if (nearby_enemy.size() > 0)
    {
        ret = nearby_enemy[0].first->get_location();
        if (side == Side::Radiant) {
            ret = pos_tup(std::get<0>(nearby_enemy[0].first->get_location()) - _dis,
                std::get<1>(nearby_enemy[0].first->get_location()) - _dis);
        }
        else {
            ret = pos_tup(std::get<0>(nearby_enemy[0].first->get_location()) + _dis,
                std::get<1>(nearby_enemy[0].first->get_location()) + _dis);
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
    PyObject* obj = Py_BuildValue("(i(dd))", decisonType::move, std::cos(a), std::sin(a));
    return obj;
}
