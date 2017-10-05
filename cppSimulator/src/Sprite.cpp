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
    if (!b_move)
        return;

    if (isAttacking())
        return;

    double dx = std::get<0>(move_target) - std::get<0>(location);
    double dy = std::get<1>(move_target) - std::get<1>(location);

    double a = atan2(dy, dx);

    if (isnan(a)) {
        printf("found nan");
        exit(-1);
    }

    if (!(std::get<0>(move_target) == 0.0
        && std::get<1>(move_target) == 0.0)) {
        double d = MovementSpeed * Engine->get_deltatick();
        if (hypot(dx, dy) < d) {
            location = move_target;
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
    if (isDead) {
        return false;
    }
    HP -= dmg;
    if (HP <= 0.0) {
        printf("I'm dead!\n");
        dead();
    }
    return true;
}

void Sprite::dead()
{
    isDead = true;
    if (NULL != v_handle) {
        PyObject_CallMethodObjArgs(
            Engine->get_canvas(), PyUnicode_FromString("delete"),
            v_handle,
            NULL
        );

        for (Sprite* s : Engine->get_sprites()) {
            if (s->side != side && S2Sdistance(*s, *this) <= 1300.0) {
                s->exp += bountyEXP;
                printf("%p get exp\n", s);
            }
        }
    }
}

double Sprite::S2Sdistance(const Sprite & s1, const Sprite & s2)
{
    double dx = std::get<0>(s1.location) - std::get<0>(s2.location);
    double dy = std::get<1>(s1.location) - std::get<1>(s2.location);
    return sqrt(dx * dx + dy * dy);
}
