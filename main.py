import sys
import math
import time
from functools import cached_property
from typing import List, Generator, Set, Type, Callable, ClassVar
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations, chain, product, groupby
from contextlib import contextmanager
from copy import copy

# Various utils
def debug(something):
    print(something, file=sys.stderr, flush=True)


def timeit(func):
    def wrapper(*arg, **kw):
        """source: http://www.daniweb.com/code/snippet368.html"""
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        debug(f"timing {t2 - t1}")
        return res

    return wrapper


# [1 2 3] => [1] [2] [3] [1, 2] [1, 3] [2, 3] [1, 2 ,3]
def all_combinations(array):
    return chain(*(list(combinations(array, i + 1)) for i in range(len(array))))


# generate all possibles and valid allocations combinations
# this is usefull to evaluate the optimal allocation of resources for a given problem
def generate_pairs_combinations(a, b) -> Generator:
    pairs = product(a, b)
    max_distinct_pairs = min(len(a), len(b))
    pairs_combinations = combinations(pairs, max_distinct_pairs)
    valid_pairs_combinations = (
        pairs_combination
        for pairs_combination in pairs_combinations
        if len(set(chain.from_iterable(pairs_combination))) == max_distinct_pairs * 2
    )
    return valid_pairs_combinations


# Data for solving
@dataclass
class Position:
    MIN_X = 0
    MIN_Y = 0
    MAX_X = 17630
    MAX_Y = 9000

    x: int
    y: int

    def distance_to(self, other) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def nearest(self, others: List["Position"]) -> "Position":
        return min(others, key=lambda other: self.distance_to(other))

    def within_bounds(self) -> bool:
        return (
            Position.MIN_X <= self.x <= Position.MAX_X
            and Position.MIN_Y <= self.y <= Position.MAX_Y
        )

    def out_of_bounds(self) -> bool:
        return not self.within_bounds()

    def with_vector(self, vector: "Vector") -> "Position":
        return Position(self.x + vector.dx, self.y + vector.dy)

    def near_any_base(self, near_distance: int) -> bool:
        return any(
            self.distance_to(base_position) <= near_distance
            for base_position in [LEFT_BASE_POSITION, RIGHT_BASE_POSITION]
        )

    def symetric_position(self) -> "Position":
        return Position(Position.MAX_X - self.x, Position.MAX_Y - self.y)


LEFT_BASE_POSITION = Position(Position.MIN_X, Position.MIN_Y)
RIGHT_BASE_POSITION = Position(Position.MAX_X, Position.MAX_Y)


@dataclass
class Vector:
    dx: int
    dy: int

    @classmethod
    def make(cls, source: Position, target: Position) -> "Vector":
        return Vector(target.x - source.x, target.y - source.y)

    @property
    def magnitude(self) -> int:
        return int(math.sqrt(self.dx ** 2 + self.dy ** 2))

    def normalize(self, norm: int) -> "Vector":
        normalization_factor = norm / self.magnitude
        return Vector(
            int(self.dx * normalization_factor), int(self.dy * normalization_factor)
        )


@dataclass
class Interception:
    starting_position: Position
    target_position: Position
    turns_to_reach: int
    turns_required: int

    @property
    def possible(self) -> bool:
        return self.turns_to_reach <= self.turns_required


@dataclass
class Path:
    positions: List[Position] = field(default_factory=list)

    def __getitem__(self, index):
        return self.positions[index]

    def __len__(self) -> int:
        return len(self.positions)

    def compute_interception(
        self, starting_position: Position, speed: int
    ) -> Interception:
        interceptions = [
            Interception(
                starting_position=starting_position,
                target_position=position,
                turns_required=position_index,
                turns_to_reach=int(position.distance_to(starting_position) / speed),
            )
            for (position_index, position) in enumerate(self.positions)
        ]
        possible_interceptions = (
            interception for interception in interceptions if interception.possible
        )
        ranked_possible_interceptions = sorted(
            possible_interceptions, key=lambda i: i.turns_required
        )
        if ranked_possible_interceptions:
            return ranked_possible_interceptions[0]
        else:
            return interceptions[0]


@dataclass
class Action:
    actor: "Hero"

    def perform(self) -> None:
        print(self.action_message() + " " + self.debug_message())

    def action_message(self) -> str:
        raise NotImplementedError

    def debug_message(self) -> str:
        return ""


@dataclass
class WaitAction(Action):
    def action_message(self) -> str:
        return "WAIT"


@dataclass
class MoveAction(Action):
    position: Position

    def action_message(self) -> str:
        return f"MOVE {self.position.x} {self.position.y}"


@dataclass
class AttackAction(MoveAction):
    target: "Monster"
    attack_type: str = "Attack"

    def debug_message(self) -> str:
        return f"{self.attack_type} {self.target.id}"


@dataclass
class KillAction(AttackAction):
    attack_type: str = "Kill"


@dataclass
class SoftenAction(AttackAction):
    attack_type: str = "Soften"


@dataclass
class FarmAction(AttackAction):
    attack_type: str = "Farm"


@dataclass
class TargetedSpellAction(Action):
    target: "Entity"

    TARGETED_SPELL_RANGE = 2200

    @property
    def legal(self) -> bool:
        return (
            self.actor.position.distance_to(self.target.position)
            <= TargetedSpellAction.TARGETED_SPELL_RANGE
        )


@dataclass
class SpellShieldAction(TargetedSpellAction):
    def action_message(self) -> str:
        return f"SPELL SHIELD {self.target.id}"


@dataclass
class SpellControlAction(TargetedSpellAction):
    target: "Entity"
    position: Position

    def action_message(self) -> str:
        return f"SPELL CONTROL {self.target.id} {self.position.x} {self.position.y}"


@dataclass
class SpellWindAction(Action):
    position: Position

    AOE_RADIUS = 1280

    def action_message(self) -> str:
        return f"SPELL WIND {self.position.x} {self.position.y}"


@dataclass
class Entity:
    id: int
    type: int
    x: int
    y: int
    shield_life: int
    is_controlled: int
    health: int
    vx: int
    vy: int
    near_base: int
    threat_for: int

    Selector = Callable[["Entity"], bool]

    @classmethod
    def deserialize(cls, str):
        args = [int(j) for j in str.split()]
        entity_type = args[1]
        if entity_type == 0:
            return Monster(*args)
        elif entity_type == 1:
            return Hero(*args)
        else:
            return Vilain(*args)

    @property
    def position(self) -> Position:
        return Position(self.x, self.y)

    @property
    def out_of_bounds(self) -> bool:
        return self.position.out_of_bounds()

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other: "Entity") -> bool:
        return self.id == other.id

    def clear_cache(self):
        for method in dir(self):
            try:
                getattr(self, method).cache_clear()
            except AttributeError:
                pass

    def is_monster(self) -> bool:
        return False

    def is_hero(self) -> bool:
        return False

    def is_vilain(self) -> bool:
        return False

    def controlled(self) -> bool:
        return bool(self.is_controlled)

    def shielded(self) -> bool:
        return bool(self.shield_life)

    def targeting_me(self) -> bool:
        return self.threat_for == 1

    def targeting_other(self) -> bool:
        return self.threat_for == 2

    def targeting_nobody(self) -> bool:
        return self.threat_for == 0

    def near_my_base(self) -> bool:
        return self.near_base == 1

    def near_other_base(self) -> bool:
        return self.near_base == 2

    def within_range(self, others: List["Entity"], radius) -> List["Entity"]:
        return [e for e in others if e.position.distance_to(self.position) <= radius]


class Hero(Entity):
    SPEED = 800
    DAMAGE = 2

    @lru_cache
    def interception(self, monster: "Monster") -> Interception:
        return monster.path.compute_interception(self.position, Hero.SPEED)

    def time_to_reach(self, monster: "Monster") -> int:
        return self.interception(monster).turns_required

    def can_reach(self, monster: "Monster") -> bool:
        return self.interception(monster).possible

    def expected_damage(self, monster: "Monster", turn: int) -> int:
        if self.can_reach(monster):
            return (turn - self.time_to_reach(monster)) * Hero.DAMAGE
        else:
            return 0

    def can_kill(self, monster: "Monster") -> bool:
        return MonsterAttack(monster, [self]).is_lethal

    def is_hero(self) -> bool:
        return True


class Vilain(Entity):
    def is_vilain(self) -> bool:
        return True


class Monster(Entity):
    TARGET_DISTANCE = 5000
    MONSTER_SPEED = 400
    # in theory 300 but theere is a small difference in my computation that mess with it
    RAID_DISTANCE = 400

    def dangerous(self) -> bool:
        return self.threat_for == 1

    def infer_move(self):
        self.x += self.vx
        self.y += self.vy

    @cached_property
    def path(self) -> Path:
        def compute_future_positions() -> Generator[Position, None, None]:
            current_position = self.position
            targetting_base = bool(self.near_base)
            current_vector = Vector(self.vx, self.vy)

            while current_position.within_bounds():
                current_position = current_position.with_vector(current_vector)
                if (
                    current_position.distance_to(LEFT_BASE_POSITION)
                    < Monster.TARGET_DISTANCE
                    and not targetting_base
                ):
                    targetting_base = True
                    current_vector = Vector.make(
                        source=current_position, target=LEFT_BASE_POSITION
                    ).normalize(Monster.MONSTER_SPEED)
                if (
                    current_position.distance_to(RIGHT_BASE_POSITION)
                    < Monster.TARGET_DISTANCE
                    and not targetting_base
                ):
                    targetting_base = True
                    current_vector = Vector.make(
                        source=current_position, target=RIGHT_BASE_POSITION
                    ).normalize(Monster.MONSTER_SPEED)

                # if current_position.within_bounds():
                yield current_position

        return Path(list(compute_future_positions()))

    def nb_turn_within_bounds(self) -> int:
        return len(self.path)

    def is_monster(self) -> bool:
        return True

    @property
    def robustness(self) -> float:
        return self.health / self.nb_turn_within_bounds()


@dataclass
class Base:
    position: Position
    health: int = 3
    mana: int = 0

    def update_from_inputs(self, str):
        self.health, self.mana = [int(j) for j in str.split()]

    @property
    def is_left(self) -> bool:
        return self.position == LEFT_BASE_POSITION

    @property
    def is_right(self) -> bool:
        return not self.is_left

    def position_for_base(self, position_relative_to_left_base: Position) -> Position:
        if self.is_left:
            return position_relative_to_left_base
        else:
            return position_relative_to_left_base.symetric_position()

    def within_range(self, entity: Entity, radius: int) -> bool:
        return self.position.distance_to(entity.position) <= radius


@dataclass
class Gamestate:
    my_base: Base
    evil_base: Base
    entities: List[Entity] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    heroes: List[Hero] = field(default_factory=list)
    vilains: List[Vilain] = field(default_factory=list)
    monsters: List[Monster] = field(default_factory=list)
    turn: int = 0

    # Accessors
    @property
    def active_heroes(self) -> List[Hero]:
        return [action.actor for action in self.actions]

    @property
    def available_heroes(self) -> List[Hero]:
        return list(
            set(self.heroes) - set(self.active_heroes) - set(self.controlled_heroes)
        )

    @property
    def controlled_heroes(self) -> List[Hero]:
        return [h for h in self.heroes if h.is_controlled]

    @property
    def entities_targetable_by_vilains(self) -> List[Entity]:
        if self.vilains:
            return [
                e
                for e in self.entities
                if min(
                    (e.position.distance_to(vilain.position) for vilain in self.vilains)
                )
                <= TargetedSpellAction.TARGETED_SPELL_RANGE
            ]
        else:
            return []

    # allow to restrict the list of heroes active
    @contextmanager
    def with_allowed_heroes(self, allowed_heroes: List[Hero]):
        original_heroes = copy(self.heroes)  # probably not necessary
        self.heroes = list(set(original_heroes) & set(allowed_heroes))
        try:
            yield
        finally:
            self.heroes = original_heroes

    @property
    def turn_actions(self):
        # The game expects first action to apply for the first hero (and so on)
        return sorted(self.actions, key=lambda a: a.actor.id)

    # New turn logic
    def begin_new_turn(self):
        self.turn += 1
        self.resolve_actions()
        self.infer_monster_positions()
        self.read_turn_inputs()
        self.cache_entities()

    def resolve_actions(self):
        # In theory, actions should be simulated on entities before inputs are received
        # This allows to have a consistent view of the state of the game even if some entities
        # ends up outside of our vision
        self.actions = []

    def infer_monster_positions(self) -> None:
        for monster in self.monsters:
            monster.infer_move()

    def read_turn_inputs(self) -> None:
        self.my_base.update_from_inputs(input())
        self.evil_base.update_from_inputs(input())
        entity_count = int(input())
        entity_inputs = [input() for _ in range(entity_count)]
        received_entities: Set[Entity] = {
            Entity.deserialize(entity_input) for entity_input in entity_inputs
        }
        self.update_entities(received_entities)

    def update_entities(self, received_entities: Set[Entity]):
        out_of_bounds_entities = {
            entity for entity in self.entities if entity.out_of_bounds
        }
        dead_entities = {
            entity
            for entity in self.entities
            if not entity.out_of_bounds and not entity in received_entities
        }
        self.entities = list(
            received_entities
            | set(self.entities) - out_of_bounds_entities - dead_entities
        )

    def cache_entities(self) -> None:
        self.heroes = [e for e in self.entities if isinstance(e, Hero)]
        self.vilains = [e for e in self.entities if isinstance(e, Vilain)]
        self.monsters = [e for e in self.entities if isinstance(e, Monster)]
        for entity in self.entities:
            entity.clear_cache()

    # misc utils
    def print_state(self) -> None:
        for entity in set(self.entities):
            debug(entity)


# Gather logic relative to an attack by a (or several hero) on a given monster
@dataclass
class MonsterAttack:
    monster: Monster
    heroes: List[Hero]
    attack_class: Type[AttackAction] = AttackAction

    @property
    def hero_count(self) -> int:
        return len(self.heroes)

    @property
    def is_lethal(self) -> bool:
        return self.max_damage >= self.monster.health

    @property
    def is_possible(self) -> bool:
        return self.max_damage > 0

    @cached_property
    def damage_profile(self) -> List[int]:
        return [
            sum(hero.expected_damage(self.monster, turn) for hero in self.heroes)
            for turn in range(self.monster.nb_turn_within_bounds())
        ]

    @property
    def max_damage(self) -> int:
        return self.damage_profile[-1]

    @property
    def time_to_reach(self) -> int:
        return next(
            turn for turn, damage in enumerate(self.damage_profile) if damage > 0
        )

    @property
    def turn_for_kill(self) -> int:
        if self.is_lethal:
            return next(
                turn
                for turn, damage in enumerate(self.damage_profile)
                if damage >= self.monster.health
            )
        else:
            return -1

    def generate_moves(
        self,
    ) -> Generator[MoveAction, None, None]:
        for hero in self.heroes:
            yield self.attack_class(
                hero, hero.interception(self.monster).target_position, self.monster
            )


# Produce actions relative to a specific objective (eg: defend base, farm, scout, etc.)
# It is acceptable to produce partial or no actions for a given gamestate if objectives are not reachable
# nor relevant
class Tactic:
    def evaluate(self, _gamestate: Gamestate) -> Generator[Action, None, None]:
        raise NotImplementedError


# Generic tactics to allocate available heroes on intercept course on target monsters
# Monsters are the primary allocation focus
class AttackMonsterTactic(Tactic):
    gamestate: Gamestate

    def evaluate(self, gamestate: Gamestate) -> Generator[Action, None, None]:
        self.gamestate = gamestate
        allocatable_heroes = gamestate.available_heroes
        for monster in self.target_monsters:
            hero_combinations = all_combinations(allocatable_heroes)
            attacks = (
                MonsterAttack(monster, heroes, self.attack_class)
                for heroes in hero_combinations
            )
            possible_attacks = [
                attack for attack in attacks if self.attack_filter(attack)
            ]
            if possible_attacks:
                best_attack = min(possible_attacks, key=lambda a: self.attack_ranker(a))
                yield from best_attack.generate_moves()
                allocatable_heroes = list(
                    set(allocatable_heroes) - set(best_attack.heroes)
                )
            else:
                # This means we can not kill with certainty the highest theat
                # we whould still try to do something about it but with other methods
                # Anyway the other heroes should not try to intercept lesser threat
                return

    @property
    def target_monsters(self) -> List[Monster]:
        raise NotImplementedError

    def attack_filter(self, attack: MonsterAttack) -> bool:
        raise NotImplementedError

    def attack_ranker(self, attack: MonsterAttack):
        raise NotImplementedError

    @property
    def attack_class(self) -> Type[AttackAction]:
        return AttackAction


class KillThreateningMonsterTactic(AttackMonsterTactic):
    @property
    def target_monsters(self) -> List[Monster]:
        # threatening priorized by time to base
        return sorted(
            (monster for monster in self.gamestate.monsters if monster.targeting_me()),
            key=lambda m: m.nb_turn_within_bounds(),
        )

    def attack_filter(self, attack: MonsterAttack) -> bool:
        return attack.is_lethal

    def attack_ranker(self, attack: MonsterAttack):
        return (attack.hero_count, attack.turn_for_kill)

    @property
    def attack_class(self) -> Type[AttackAction]:
        return KillAction


class SoftenThreateningMonsterTactic(AttackMonsterTactic):
    @property
    def target_monsters(self) -> List[Monster]:
        # threatening priorized by time to base
        targetted_monsters = {
            action.target
            for action in self.gamestate.actions
            if isinstance(action, AttackAction)
        }
        threatening_monsters = {
            monster for monster in self.gamestate.monsters if monster.targeting_me()
        }
        targetable_monsters = list(threatening_monsters - targetted_monsters)
        return sorted(
            targetable_monsters,
            key=lambda m: m.nb_turn_within_bounds(),
        )

    def attack_filter(self, attack: MonsterAttack) -> bool:
        return attack.is_possible

    def attack_ranker(self, attack: MonsterAttack):
        # otherwise we select the min damage first
        return -attack.max_damage

    @property
    def attack_class(self) -> Type[AttackAction]:
        return SoftenAction


# monster farm should find the path that maximize the mana gained
# each hero do a step in direction of a monster
# a* with heuristic
# hero, monsters,


# compute overlap on a given turn (if i dont remove monster, this is easy, if i do i must simulate) if i simulate the


# Tactics intended to produce the maximum mana
# Hero are the primary focus
class FarmTactic(Tactic):
    gamestate: Gamestate

    def evaluate(self, gamestate: Gamestate) -> Generator[Action, None, None]:
        self.gamestate = gamestate
        allocatable_heroes = gamestate.available_heroes
        targetable_monsters = self.target_monsters
        attack_pairs = product(allocatable_heroes, targetable_monsters)
        attacks = (
            MonsterAttack(monster, [hero], FarmAction) for hero, monster in attack_pairs
        )
        possible_attacks = (attack for attack in attacks if self.attack_filter(attack))
        ranked_attacks = sorted(possible_attacks, key=lambda a: self.attack_ranker(a))
        picked_heroes = []
        farmed_monsters = []
        for attack in ranked_attacks:
            if (
                attack.heroes[0] not in picked_heroes
                and attack.monster not in farmed_monsters
            ):
                yield from attack.generate_moves()
                picked_heroes.append(attack.heroes[0])
                farmed_monsters.append(attack.monster)

    @property
    def target_monsters(self) -> List[Monster]:
        raise NotImplementedError

    def attack_filter(self, attack: MonsterAttack) -> bool:
        raise NotImplementedError

    def attack_ranker(self, attack: MonsterAttack):
        raise NotImplementedError


class FarmNearestMonsterTactic(FarmTactic):
    @property
    def target_monsters(self) -> List[Monster]:
        # threatening priorized by time to base
        targetted_monsters = {
            action.target
            for action in self.gamestate.actions
            if isinstance(action, AttackAction)
        }
        targetable_monsters = list(set(gamestate.monsters) - targetted_monsters)
        return targetable_monsters

    def attack_filter(self, attack: MonsterAttack) -> bool:
        return attack.is_possible

    def attack_ranker(self, attack: MonsterAttack):
        return attack.time_to_reach


# Try to reach a given area
@dataclass
class GuardAreaTactic(Tactic):
    positions: List[Position]
    tolerance: int
    gamestate: Gamestate = field(init=False)

    def evaluate(self, gamestate: Gamestate) -> Generator[Action, None, None]:
        self.gamestate = gamestate
        guard_positions = self.guard_positions

        for allocatable_hero in gamestate.available_heroes:
            nearest_guard_position = allocatable_hero.position.nearest(guard_positions)
            guard_positions.remove(nearest_guard_position)
            if nearest_guard_position.distance_to(allocatable_hero) >= self.tolerance:
                yield MoveAction(allocatable_hero, nearest_guard_position)

    @property
    def guard_positions(self) -> List[Position]:
        return [self.base.position_for_base(position) for position in self.positions]

    @property
    def base(self) -> Base:
        return self.gamestate.my_base


# Attempt to cast spell on the maximum of eligible targets with the current available heroes
# The primary focus is the number of targets allocated
# (since the priority on entites can be ajusted externally)
@dataclass
class TargetedSpellTactic(Tactic):
    target_selector: Entity.Selector
    mana_threshold: int

    def evaluate(self, gamestate: Gamestate) -> Generator[Action, None, None]:
        self.gamestate = gamestate
        eligible_targets = {
            entity
            for entity in gamestate.entities
            if self.target_selector(entity)
            and not entity.shielded()
            and not self.target_restriction(entity)
        }
        already_targeted_entities = {
            spell.target
            for spell in gamestate.actions
            if isinstance(spell, TargetedSpellAction)
        }
        selectable_targets = list(eligible_targets - already_targeted_entities)
        allocatable_heroes = gamestate.available_heroes
        # TODO mana allocation
        if gamestate.my_base.mana > self.mana_threshold:
            spell_pairs = product(allocatable_heroes, selectable_targets)
            spells = (
                self.make_spell(hero, target, gamestate) for hero, target in spell_pairs
            )
            legal_spells = (spell for spell in spells if spell.legal)
            spells_grouped_by_hero = [
                list(spells)
                for _, spells in groupby(legal_spells, key=lambda s: s.actor)
            ]
            spells_by_hero_ascending = sorted(spells_grouped_by_hero, key=len)
            picked_targets = []
            for spells in spells_by_hero_ascending:
                valid_spell = next(
                    (spell for spell in spells if spell.target not in picked_targets),
                    None,
                )
                if valid_spell:
                    yield valid_spell
                    picked_targets.append(valid_spell.target)

    def make_spell(
        self, hero: Hero, target: Entity, gamestate: Gamestate
    ) -> TargetedSpellAction:
        raise NotImplementedError

    def target_restriction(self, entity: Entity) -> bool:
        return False


class SendMonsterToOtherBaseTactic(TargetedSpellTactic):
    def make_spell(
        self, hero: Hero, target: Entity, gamestate: Gamestate
    ) -> TargetedSpellAction:
        return SpellControlAction(hero, target, gamestate.evil_base.position)

    def target_restriction(self, entity: Entity) -> bool:
        return not entity.is_monster() or entity.near_my_base() or entity.controlled()


class PassiveTactic(Tactic):
    def evaluate(self, gamestate: Gamestate) -> Generator[Action, None, None]:
        for hero in gamestate.controlled_heroes:
            yield (WaitAction(hero))


class ShieldSelfTactic(Tactic):
    opfor_use_mindcontrol: ClassVar[bool] = False

    def evaluate(self, gamestate: Gamestate) -> Generator[Action, None, None]:
        entities_within_vilain_control_range = gamestate.entities_targetable_by_vilains
        vulnerable_heroes = (
            hero
            for hero in gamestate.available_heroes
            if not hero.shielded() and hero in entities_within_vilain_control_range
        )
        ShieldSelfTactic.opfor_use_mindcontrol = (
            ShieldSelfTactic.opfor_use_mindcontrol
            or any(hero for hero in gamestate.heroes if hero.controlled())
        )
        if ShieldSelfTactic.opfor_use_mindcontrol:
            for hero in vulnerable_heroes:
                yield (SpellShieldAction(hero, hero))


class ShieldThreateningMonsterTactic(Tactic):
    def evaluate(self, gamestate: Gamestate) -> Generator[Action, None, None]:
        if not gamestate.available_heroes:
            return
        # in theory there may be several but in practice this is a mono strategy
        # if I need another hero, I'll add an allocation logic
        eligible_hero = gamestate.available_heroes[0]
        debug(gamestate.monsters)
        eligible_monsters = [
            monster
            for monster in gamestate.monsters
            if monster.targeting_other()
            # and monster.near_other_base()
            and not monster.shielded() and monster.health > 16
        ]
        debug(eligible_monsters)
        ranked_eligible_monsters = sorted(
            eligible_monsters, key=lambda m: m.robustness, reverse=True
        )
        debug(ranked_eligible_monsters)
        best_target = next(
            (
                monster
                for monster in ranked_eligible_monsters
                if monster.within_range(
                    [eligible_hero], TargetedSpellAction.TARGETED_SPELL_RANGE
                )
            ),
            None,
        )
        if best_target:
            yield SpellShieldAction(eligible_hero, best_target)


class DefensiveWindTactic(Tactic):
    def evaluate(self, gamestate: Gamestate) -> Generator[Action, None, None]:
        dangerous_eligible_monsters: Generator[Monster, None, None] = (
            e
            for e in gamestate.entities
            if e.is_monster()
            and gamestate.my_base.within_range(e, 3000)
            and e.targeting_me()
            and not e.shielded()
        )
        # actually several hero may have killed it but this is dangerous because a wind could seal the deal
        # we may use slightly more wind that required but better be safe than sorry
        unkillable_eligible_monsters = [
            m
            for m in dangerous_eligible_monsters
            if not any(hero.can_kill(m) for hero in gamestate.available_heroes)
        ]
        potential_savior_heroes = [
            hero
            for hero in gamestate.available_heroes
            if any(
                hero.within_range(
                    unkillable_eligible_monsters, SpellWindAction.AOE_RADIUS
                )
            )
        ]
        if potential_savior_heroes:
            best_savior = max(
                potential_savior_heroes,
                key=lambda h: len(
                    h.within_range(
                        unkillable_eligible_monsters, SpellWindAction.AOE_RADIUS
                    )
                ),
            )
            yield SpellWindAction(
                best_savior,
                gamestate.evil_base.position,
            )


# Entity selector builder
class Esb:
    @staticmethod
    def _and(*selectors: Entity.Selector) -> Entity.Selector:
        def combined_selectors(entity) -> bool:
            return all(selector(entity) for selector in selectors)

        return combined_selectors

    @staticmethod
    def min_health(health: int) -> Entity.Selector:
        def selector(entity) -> bool:
            return entity.health >= health

        return selector

    @staticmethod
    def _not(selector: Entity.Selector) -> Entity.Selector:
        def opposite_selector(entity) -> bool:
            return not selector(entity)

        return opposite_selector


# Meta tactics
ControlThreateningMonsterTactic = SendMonsterToOtherBaseTactic(
    target_selector=Esb._and(Esb.min_health(12), Entity.targeting_me),
    mana_threshold=30,
)
ControlNeutralMonsterTactic = SendMonsterToOtherBaseTactic(
    target_selector=Esb._and(
        Esb.min_health(16),
        Entity.targeting_nobody,
    ),
    mana_threshold=100,
)

# A role represents a set of Rules an actor should follow.
# Each rule is evaluated sequentially until all allocated heroes are staffed.
# Same roles are evaluated simultenaously.
# Targets and objectives allocation are therefore shared and can be optimized.
#
# Roles are allocated to heroes based on proximity according to its specified position
@dataclass
class Role:
    position: Position
    leeway: int = 2000

    @property
    def tactics(self) -> List[Tactic]:
        raise NotImplementedError

    # what the hero should do if none actions are applicable
    @property
    def idle_tactic(self) -> Tactic:
        return FarmNearestMonsterTactic()

    def __hash__(self) -> int:
        return hash(self.__class__.__name__ + f"{self.position}")


class DefenderRole(Role):
    TOP_POSITION = Position(7200, 2000)
    MIDDLE_POSITION = Position(5600, 4900)
    BOTTOM_POSITION = Position(2600, 6200)
    INNER_POSITION = Position(3000, 2500)

    @property
    def tactics(self) -> List[Tactic]:
        return [
            ShieldSelfTactic(),
            DefensiveWindTactic(),
            ControlThreateningMonsterTactic,
            ControlNeutralMonsterTactic,
            KillThreateningMonsterTactic(),
            SoftenThreateningMonsterTactic(),
        ]


class DisruptorRole(Role):
    MIDDLE_POSITION = Position(13000, 6000)

    @property
    def tactics(self) -> List[Tactic]:
        return [
            ShieldThreateningMonsterTactic(),
            ControlNeutralMonsterTactic,
        ]


@dataclass
class Mission:
    role: Role
    hero: Hero
    my_base: Base

    @property
    def fitness(self) -> int:
        return 20000 - int(
            self.hero.position.distance_to(
                self.my_base.position_for_base(self.role.position)
            )
        )


# Produce actions for the current turn of a given gamestate by defining the applicable tactics set and how
# they are prioritized
class Strategy:
    # Define this turn gamestate actions
    def apply(self, gamestate: Gamestate):
        raise NotImplementedError


# Eval each tactics sequentially until all heroes are staffed
# Once an hero has been assigned, it can't be overrided by another downstream tactic
@dataclass
class PriorityStrategy(Strategy):
    tactics: List[Tactic] = field(default_factory=list)

    def apply(self, gamestate: Gamestate):
        for tactic in self.tactics:
            actions = list(tactic.evaluate(gamestate))
            gamestate.actions += actions


@dataclass
class RoleBasedStrategy(Strategy):
    roles: List[Role]

    def apply(self, gamestate: Gamestate):
        missions = self.allocate_missions(gamestate)
        grouped_missions_by_role = groupby(missions, key=lambda m: m.role.__class__)
        for _, missions in grouped_missions_by_role:
            missions = list(missions)
            with gamestate.with_allowed_heroes([mission.hero for mission in missions]):
                PriorityStrategy(self.tactics_for(missions)).apply(gamestate)

    def allocate_missions(self, gamestate: Gamestate) -> List[Mission]:
        pairs_combinations = generate_pairs_combinations(gamestate.heroes, self.roles)
        missions_combinations = (
            [Mission(hero, role, gamestate.my_base) for role, hero in pairs_combination]
            for pairs_combination in pairs_combinations
        )
        best_mission_combination = max(
            missions_combinations,
            key=lambda mission_combination: sum(
                mission.fitness for mission in mission_combination
            ),
        )
        return best_mission_combination

    def tactics_for(self, missions: List[Mission]) -> List[Tactic]:
        roles = [mission.role for mission in missions]
        role_positions = [role.position for role in roles]
        role = roles[0]
        return [
            *role.tactics,
            GuardAreaTactic(role_positions, role.leeway),
            role.idle_tactic,
            GuardAreaTactic(role_positions, 0),
            PassiveTactic(),
        ]


basic_strategy = RoleBasedStrategy(
    roles=[
        DefenderRole(DefenderRole.MIDDLE_POSITION),
        DefenderRole(DefenderRole.TOP_POSITION),
        DefenderRole(DefenderRole.BOTTOM_POSITION),
    ]
)

late_strategy = RoleBasedStrategy(
    roles=[
        DefenderRole(DefenderRole.INNER_POSITION, 2000),
        DefenderRole(DefenderRole.MIDDLE_POSITION, 1500),
        DisruptorRole(DisruptorRole.MIDDLE_POSITION, 1500),
    ]
)


def init_gamestate():
    base_x, base_y = [int(i) for i in input().split()]
    heroes_per_player = int(input())
    my_base_position = Position(base_x, base_y)
    my_base = Base(position=Position(base_x, base_y))
    evil_base = Base(position=my_base.position.symetric_position())
    return Gamestate(my_base, evil_base)


# game loop
gamestate = init_gamestate()
while True:
    gamestate.begin_new_turn()
    # gamestate.print_state()
    if gamestate.turn <= 80:
        strategy = basic_strategy
    else:
        strategy = late_strategy
    strategy.apply(gamestate)
    for action in gamestate.turn_actions:
        debug(action)
        action.perform()


# soften should allocate only one hero (more if there are more heroes)
# farm should attempt to target cluster
# farm should be more intelligent (rank controlled target as last possible target)
# scout strategy
