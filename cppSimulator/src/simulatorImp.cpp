#include "simulatorImp.h"
#include "Config.h"
#include "Event.h"
#include "Tower.h"

#include <algorithm>

cppSimulatorImp::cppSimulatorImp(cppSimulatorObject* obj, PyObject* canvas)
    :self(obj), tick_time(0.0), tick_per_second(Config::tick_per_second), 
    delta_tick(1.0 / Config::tick_per_second), canvas(canvas)
{
    Py_INCREF(self);
    Py_XINCREF(canvas);
    EventFactory::CreateSpawnEvnt(this);
    Tower::initTowers(this);
}

cppSimulatorImp::~cppSimulatorImp()
{
    for (auto p : allSprites) {
        delete p;
    }
    Py_DECREF(self);
    Py_XDECREF(canvas);
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
        if (s->get_side() != sprite->get_side()) {
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
