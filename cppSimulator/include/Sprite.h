#ifndef __SPRITE_H__
#define __SPRITE_H__

#include "Config.h"

//forward decl
class cppSimulatorImp;

class Sprite {
public:
    Sprite(cppSimulatorImp* Engine,
        PyObject* canvas,
        Side side, pos_tup loc, double HP,
        double MP, double Speed,double Armor,
        double ATK,double ATKRange,double SightRange,
        double Bounty,double bountyEXP,
        double BAT,double AS):
        Engine(Engine),canvas(canvas),side(side),
        location(loc),HP(HP),MP(MP),MovementSpeed(Speed),
        BaseAttackTime(BAT),AttackSpeed(AS),Armor(Armor),
        Attack(ATK),AttackRange(ATKRange){}
protected:
    cppSimulatorImp* Engine;
    PyObject* canvas;
    Side side;
    pos_tup location;
    double HP;
    double MP;
    double MovementSpeed;
    double BaseAttackTime;
    double AttackSpeed;
    double Armor;
    double Attack;
    double AttackRange;
    double SightRange;
    double Bounty;
    double bountyEXP;
    double LastAttackTime;
    double AttackTime;
    double exp;
    double move_target;
    PyObject* v_handle;
    bool isDead;
};

#endif//__SPRITE_H__