#include "simulatorImp.h"
#include "Config.h"
#include "Event.h"
#include "Tower.h"
#include "Hero.h"
#include "log.h"

#include <algorithm>

cppSimulatorImp::cppSimulatorImp(cppSimulatorObject* obj, PyObject* canvas)
    :self(obj), tick_time(0.0), tick_per_second(Config::tick_per_second), 
    delta_tick(1.0 / Config::tick_per_second), canvas(canvas)
{
    EventFactory::CreateSpawnEvnt(this);
    Tower::initTowers(this);
    Hero* r_hero = new Hero(this, Side::Radiant, "ShadowFiend");
    Hero* d_hero = new Hero(this, Side::Dire, "ShadowFiend");
    addSprite(r_hero);
    addSprite(d_hero);
    RadiantHeros.push_back(r_hero);
    DireHeros.push_back(d_hero);
}

cppSimulatorImp::~cppSimulatorImp()
{
    std::vector<Event> last_events;
    while (!queue.empty())
    {
        auto event = queue.top();
        queue.pop();
        last_events.push_back(event);
    }

    for (auto& e : last_events)
    {
        e.activate();
    }
    
    for (auto p : allSprites) {
        delete p;
    }
}

double cppSimulatorImp::get_time()
{
    return tick_time;
}

void cppSimulatorImp::loop()
{
    tick_tick();
    while (!queue.empty() &&
        queue.top().get_time() < tick_time)
    {
        auto event = queue.top();
        queue.pop();
        event.activate();
    }

    for (Sprite* s : Sprites) {
        s->step();
    }

    for (Sprite* s : Sprites) {
        s->move();
    }

    for (Sprite* s : Sprites) {
        s->draw();
    }

    for (auto it = Sprites.begin(); it != Sprites.end();) {
        if ((*it)->isDead()) {
            it = Sprites.erase(it);
        }
        else {
            ++it;
        }
    }
}

std::vector<std::pair<Sprite*, double>> cppSimulatorImp::get_nearby_enemy(Sprite * sprite)
{
    std::vector<std::pair<Sprite*, double>> ret;
    for (Sprite* s : Sprites) {
        if (s->get_side() != sprite->get_side() && s != sprite) {
            double d = Sprite::S2Sdistance(*s, *sprite);
            if (d < sprite->get_SightRange()) {
                ret.push_back(std::make_pair(s, d));
            }
        }
    }
    auto sort_fn = [](const std::pair<Sprite*, double>& l, const std::pair<Sprite*, double>&r)->bool {
        return l.second < r.second;
    };
    std::sort(ret.begin(), ret.end(), sort_fn);
    return ret;
}

std::vector<std::pair<Sprite*, double>> cppSimulatorImp::get_nearby_enemy(Sprite * sprite,std::function<bool(Sprite*)> filter)
{
    std::vector<std::pair<Sprite*, double>> ret;
    for (Sprite* s : Sprites) {
        if (s->get_side() != sprite->get_side() && s != sprite && filter(s)) {
            double d = Sprite::S2Sdistance(*s, *sprite);
            if (d < sprite->get_SightRange()) {
                ret.push_back(std::make_pair(s, d));
            }
        }
    }
    auto sort_fn = [](const std::pair<Sprite*, double>& l, const std::pair<Sprite*, double>&r)->bool {
        return l.second < r.second;
    };
    std::sort(ret.begin(), ret.end(), sort_fn);
    return ret;
}

std::vector<std::pair<Sprite*, double>> cppSimulatorImp::get_nearby_ally(Sprite * sprite)
{
    std::vector<std::pair<Sprite*, double>> ret;
    for (Sprite* s : Sprites) {
        if (s->get_side() == sprite->get_side()
            && s != sprite) {
            double d = Sprite::S2Sdistance(*s, *sprite);
            if (d < sprite->get_SightRange()) {
                ret.push_back(std::make_pair(s, d));
            }
        }
    }
    auto sort_fn = [](const std::pair<Sprite*, double>& l, const std::pair<Sprite*, double>&r)->bool {
        return l.second < r.second;
    };
    std::sort(ret.begin(), ret.end(), sort_fn);
    return ret;
}

std::vector<std::pair<Sprite*, double>> cppSimulatorImp::get_nearby_ally(Sprite * sprite, std::function<bool(Sprite*)> filter)
{
    std::vector<std::pair<Sprite*, double>> ret;
    for (Sprite* s : Sprites) {
        if (s->get_side() == sprite->get_side()
            && s != sprite && filter(s)) {
            double d = Sprite::S2Sdistance(*s, *sprite);
            if (d < sprite->get_SightRange()) {
                ret.push_back(std::make_pair(s, d));
            }
        }
    }
    auto sort_fn = [](const std::pair<Sprite*, double>& l, const std::pair<Sprite*, double>&r)->bool {
        return l.second < r.second;
    };
    std::sort(ret.begin(), ret.end(), sort_fn);
    return ret;
}

void cppSimulatorImp::set_order(PyObject * args, PyObject * kwds)
{
    char* side = NULL;
    int idx = 0;
    PyObject* order;
    if (!PyArg_ParseTuple(args, "siO", &side, &idx, &order)) {
        LOG << "Parse Arg error";
        Logger::getInstance().flush();
        exit(2);
    }
    if (0 == strcmp(side,"Radiant")) {
        RadiantHeros[idx]->set_order(order);
    }
    else {
        DireHeros[idx]->set_order(order);
    }
}

PyObject* cppSimulatorImp::get_state_tup(std::string side, int idx)
{
    if (side == "Radiant") {
        return RadiantHeros[idx]->get_state_tup();
    }
    else {
        return DireHeros[idx]->get_state_tup();
    }
}

PyObject* cppSimulatorImp::predefined_step(std::string side, int idx) 
{
    if (side == "Radiant") {
        return RadiantHeros[idx]->predefined_step();
    }
    else {
        return DireHeros[idx]->predefined_step();
    }
}
