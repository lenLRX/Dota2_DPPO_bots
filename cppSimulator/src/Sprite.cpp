#include "Sprite.h"
#include "simulatorImp.h"
#include "Event.h"

#include <cmath>
#include <cstdlib>

void Sprite::attack(Sprite* target)
{
    LastAttackTime = Engine->get_time();
    EventFactory::CreateAttackEvnt(this, target);
}

bool Sprite::isAttacking()
{
    return (Engine->get_time() - LastAttackTime)
        < AttackTime;
}

void Sprite::move()
{
    if (isDead())
        return;

    if (!b_move)
        return;

    if (isAttacking()) {
        b_move = false;
        return;
    }
        

    double dx = std::get<0>(move_target) - std::get<0>(location);
    double dy = std::get<1>(move_target) - std::get<1>(location);
    
    if(dx == 0.0 && dy == 0.0){
        return;
    }

    double a = std::atan2(dy, dx);

    if (std::isnan(a)) {
        printf("found nan\n");
        fflush(stdout);
        exit(-1);
    }

    if (!(std::get<0>(move_target) == 0.0
        && std::get<1>(move_target) == 0.0)) {
        double d = MovementSpeed * Engine->get_deltatick();
        if (hypot(dx, dy) < d) {
            location = move_target;
            b_move = false;
        }
        else {
            location = pos_tup(std::get<0>(location) + d * cos(a),
                std::get<1>(location) + d * sin(a));
        }
    }

    if (std::get<0>(location) > Config::bound_length) {
        location = pos_tup(Config::bound_length, std::get<1>(location));
    }
    if (std::get<1>(location) > Config::bound_length) {
        location = pos_tup(std::get<0>(location), Config::bound_length);
    }
    if (std::get<0>(location) < -Config::bound_length) {
        location = pos_tup(-Config::bound_length, std::get<1>(location));
    }
    if (std::get<1>(location) < -Config::bound_length) {
        location = pos_tup(std::get<0>(location), -Config::bound_length);
    }
}

bool Sprite::damadged(double dmg)
{
    if (isDead()) {
        return false;
    }
    HP -= dmg;
    if (HP <= 0.0) {
        dead();
    }
    return true;
}

void Sprite::dead()
{
    _isDead = true;
    remove_visual_ent();
    for (Sprite* s : Engine->get_sprites()) {
        if (s->side != side && S2Sdistance(*s, *this) <= 1300.0) {
            s->exp += bountyEXP;
        }
    }
}

void Sprite::remove_visual_ent()
{
    if (NULL != v_handle) {
        PyObject* delete_fn = PyObject_GetAttrString(canvas, "delete");
        PyObject* args = Py_BuildValue("(O)", v_handle);
        PyObject* kwargs = Py_BuildValue("{}");
        Py_XDECREF(PyObject_Call(delete_fn, args, kwargs));
        Py_DECREF(kwargs);
        Py_DECREF(args);
        Py_DECREF(delete_fn);
        Py_DECREF(v_handle);
        v_handle = NULL;
    }
}

double Sprite::S2Sdistance(const Sprite & s1, const Sprite & s2)
{
    double dx = std::get<0>(s1.location) - std::get<0>(s2.location);
    double dy = std::get<1>(s1.location) - std::get<1>(s2.location);
    return sqrt(dx * dx + dy * dy);
}
