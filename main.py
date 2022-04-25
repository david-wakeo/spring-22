import sys
import math
import time
from functools import cached_property
from typing import List, Generator, Set, Type
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations, chain, product

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
        return Position(
            Position.MAX_X - self.x,
            Position.MAX_Y - self.y
        )


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
        return int(math.sqrt(self.dx**2 + self.dy**2))

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
class SpellShieldAction(Action):
    target: "Entity"

    def action_message(self) -> str:
        return f"SPELL SHIELD {self.target.id}"


@dataclass
class SpellControlAction(Action):
    target: "Entity"
    position: Position

    def action_message(self) -> str:
        return f"SPELL CONTROL {self.target.id} {self.position.x} {self.position.y}"


@dataclass
class SpellWindAction(Action):
    position: Position

    def action_message(self) -> str:
        return f"SPELL WIND #{self.position.x} {self.position.y}"


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


class Vilain(Entity):
    pass


class Monster(Entity):
    TARGET_DISTANCE = 5000
    MONSTER_SPEED = 400
    # in theory 300 but theere is a small difference in my computation that mess with it
    RAID_DISTANCE = 400

    @property
    def is_dangerous(self):
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

            while current_position.within_bounds() or (
                targetting_base
                and not current_position.near_any_base(Monster.RAID_DISTANCE)
            ):
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


@dataclass
class Base:
    position: Position
    health: int = 3
    mana: int = 0

    def infer_opposite_base(self) -> "Base":
        if self.is_right:
            return Base(RIGHT_BASE_POSITION)
        else:
            return Base(LEFT_BASE_POSITION)

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



@dataclass
class Gamestate:
    my_base: Base
    evil_base: Base
    entities: List[Entity] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    heroes: List[Hero] = field(default_factory=list)
    vilains: List[Vilain] = field(default_factory=list)
    monsters: List[Monster] = field(default_factory=list)

    # Accessors
    @property
    def active_heroes(self) -> List[Hero]:
        return [action.actor for action in self.actions]

    @property
    def available_heroes(self):
        return [h for h in self.heroes if h not in self.active_heroes]

    @property
    def turn_actions(self):
        # The game expects first action to apply for the first hero (and so on)
        return sorted(self.actions, key=lambda a: a.actor.id)

    # New turn logic
    def begin_new_turn(self):
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
        for i in range(2):
            health, mana = [int(j) for j in input().split()]
        entity_count = int(input())
        entity_inputs = [input() for i in range(entity_count)]
        received_entities = {
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
        return next(turn for turn, damage in enumerate(self.damage_profile) if damage > 0)

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
                hero,
                hero.interception(self.monster).target_position,
                self.monster
            )

# Produce actions relative to a specific objective (eg: defend base, farm, scout, etc.)
# It is acceptable to produce partial or no actions for a given gamestate if objectives are not reachable
# nor relevant
class Tactic:
    def evaluate(self, gamestate: Gamestate) -> Generator[Action, None, None]:
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
            attacks = (MonsterAttack(monster, heroes, self.attack_class) for heroes in hero_combinations)
            possible_attacks = [
                attack for attack in attacks if self.attack_filter(attack)
            ]
            if possible_attacks:
                best_attack = min(possible_attacks, key=lambda a: self.attack_ranker(a))
                yield from best_attack.generate_moves()
                allocatable_heroes = list(
                    set(allocatable_heroes) - set(best_attack.heroes)
                )

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
            (monster for monster in self.gamestate.monsters if monster.is_dangerous),
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
            action.target for action in self.gamestate.actions if isinstance(action, AttackAction)
        }
        threatening_monsters = { monster for monster in self.gamestate.monsters if monster.is_dangerous }
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

# Tactics intended to produce the maximum mana
# Hero are the primary focus
class FarmTactic(Tactic):
    gamestate: Gamestate

    def evaluate(self, gamestate: Gamestate) -> Generator[Action, None, None]:
        self.gamestate = gamestate
        allocatable_heroes = gamestate.available_heroes
        targetable_monsters = self.target_monsters
        attack_pairs = product(allocatable_heroes, targetable_monsters)
        attacks = (MonsterAttack(
            monster, [hero], FarmAction
        ) for hero, monster in attack_pairs)
        possible_attacks = (attack for attack in attacks if self.attack_filter(attack))
        ranked_attacks = sorted(possible_attacks, key=lambda a: self.attack_ranker(a))
        picked_heroes = []
        farmed_monsters = []
        for attack in ranked_attacks:
            if attack.heroes[0] not in picked_heroes and attack.monster not in farmed_monsters:
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
            action.target for action in self.gamestate.actions if isinstance(action, AttackAction)
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
    gamestate: Gamestate = field(init = False)

    BASE_OUTSKIRT_POSITIONS = [
        Position(7200, 2000),
        Position(5600, 4900),
        Position(2600, 6200)
    ]

    def evaluate(self, gamestate: Gamestate) -> Generator[Action, None, None]:
        self.gamestate = gamestate
        debug(self.positions)
        debug(self.guard_positions)
        guard_positions = self.guard_positions

        for allocatable_hero in gamestate.available_heroes:
            nearest_guard_position = allocatable_hero.position.nearest(guard_positions)
            guard_positions.remove(nearest_guard_position)
            if nearest_guard_position.distance_to(allocatable_hero) >= self.tolerance:
                yield MoveAction(allocatable_hero, nearest_guard_position)

    @property
    def guard_positions(self) -> List[Position]:
        return [
            self.base.position_for_base(
                position
            ) for position in self.positions
        ]

    @property
    def base(self) -> Base:
        return self.gamestate.my_base


class PassiveTactic(Tactic):
    def evaluate(self, gamestate: Gamestate) -> Generator[Action, None, None]:
        for hero in gamestate.available_heroes:
            yield (WaitAction(hero))


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


def init_gamestate():
    base_x, base_y = [int(i) for i in input().split()]
    heroes_per_player = int(input())
    my_base_position = Position(base_x, base_y)
    my_base = Base(position=Position(base_x, base_y))
    evil_base = my_base.infer_opposite_base()
    return Gamestate(my_base, evil_base)


# Strategy definition
# Defensive strategy aiming to maximize mana and map awareness as second objective.
# This may be enough against bad bot otherwise this is an early game strategy setting fundation for late
# game strategies
farm_strategy = PriorityStrategy(
    tactics=[
        KillThreateningMonsterTactic(),
        SoftenThreateningMonsterTactic(),
        GuardAreaTactic(GuardAreaTactic.BASE_OUTSKIRT_POSITIONS, 2000),
        FarmNearestMonsterTactic(),
        GuardAreaTactic(GuardAreaTactic.BASE_OUTSKIRT_POSITIONS, 0),
        PassiveTactic(),
    ]
)

# game loop
gamestate = init_gamestate()
while True:
    gamestate.begin_new_turn()
    # gamestate.print_state()
    farm_strategy.apply(gamestate)
    for action in gamestate.turn_actions:
        debug(action)
        action.perform()
