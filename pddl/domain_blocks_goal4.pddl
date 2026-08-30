(define (domain blocks-goal4)
  (:requirements :strips :typing)
  (:types block position)

  (:predicates
    (ontable ?x - block)
    (on ?x - block ?y - block)
    (clear ?x - block)
    (holding ?x - block)
    (handempty)
    (at-position ?x - block ?p - position)
    (position-free ?p - position)
    (allowed-position ?x - block ?p - position)  ; Constraint: block X can only be placed at position P
    (table-position ?p - position)               ; Marks positions where blocks can be placed on table
    (position-above ?p-top - position ?p-bottom - position)  ; P-top is directly above P-bottom
  )

  ;; Pickup from specific position - position-aware
  ;; For blocks at designated positions with allowed-position constraints
  (:action pickup-at
    :parameters (?x - block ?p - position)
    :precondition (and
      (ontable ?x)
      (clear ?x)
      (handempty)
      (at-position ?x ?p)
    )
    :effect (and
      (holding ?x)
      (not (ontable ?x))
      (not (clear ?x))
      (not (handempty))
      (not (at-position ?x ?p))
      (position-free ?p)
    )
  )

  ;; Fallback: Simple pickup for blocks not at designated positions
  ;; Used when blocks from other goals are in the way
  (:action pickup
    :parameters (?x - block)
    :precondition (and
      (ontable ?x)
      (clear ?x)
      (handempty)
    )
    :effect (and
      (holding ?x)
      (not (ontable ?x))
      (not (clear ?x))
      (not (handempty))
    )
  )

  ;; Put down at specific position
  (:action putdown-at
    :parameters (?x - block ?p - position)
    :precondition (and
      (holding ?x)
      (position-free ?p)
      (allowed-position ?x ?p)  ; Only allow placing block at its designated position
      (table-position ?p)        ; Only allow placing on table positions (not stacking positions)
    )
    :effect (and
      (ontable ?x)
      (clear ?x)
      (handempty)
      (not (holding ?x))
      (at-position ?x ?p)
      (not (position-free ?p))
    )
  )

  ;; Fallback: put a held block down on any free patch of table (no named position).
  ;;
  ;; Without this the domain deadlocks on recovery: a block placed slightly off its
  ;; named position reads as plain (on ?x ?y) / (ontable ?x) with no at-position fact,
  ;; and once the planner unstacks such a block the hand can never open again --
  ;; putdown-at demands an allowed table position (a _top block has none) and stack-at
  ;; demands the support already be at-position (impossible while the hand is full).
  ;; Observed: y4 on y2, both ~6 mm off pos_r1_c3, made the whole problem unsolvable
  ;; under both EHC and A*. The physical robot can always put a block on an empty
  ;; patch of table; the executive grounds this with find_free_table_spot().
  (:action putdown
    :parameters (?x - block)
    :precondition (and
      (holding ?x)
    )
    :effect (and
      (ontable ?x)
      (clear ?x)
      (handempty)
      (not (holding ?x))
    )
  )

  ;; Unstack from specific position - position-aware
  ;; For blocks at designated positions with allowed-position constraints
  (:action unstack-at
    :parameters (?x - block ?y - block ?p - position)
    :precondition (and
      (on ?x ?y)
      (clear ?x)
      (handempty)
      (at-position ?x ?p)
    )
    :effect (and
      (holding ?x)
      (clear ?y)
      (not (on ?x ?y))
      (not (clear ?x))
      (not (handempty))
      (not (at-position ?x ?p))
      (position-free ?p)
    )
  )

  ;; Fallback: Simple unstack for blocks not at designated positions
  ;; Used when blocks from other goals are in the way
  (:action unstack
    :parameters (?x - block ?y - block)
    :precondition (and
      (on ?x ?y)
      (clear ?x)
      (handempty)
    )
    :effect (and
      (holding ?x)
      (clear ?y)
      (not (on ?x ?y))
      (not (clear ?x))
      (not (handempty))
    )
  )

  ;; Stack on block at position
  ;; ?p-bottom is the position of the bottom block ?y
  ;; ?p-top is the position where the top block ?x will be
  (:action stack-at
    :parameters (?x - block ?y - block ?p-bottom - position ?p-top - position)
    :precondition (and
      (holding ?x)
      (clear ?y)
      (at-position ?y ?p-bottom)
      (allowed-position ?x ?p-top)  ; Top block must be allowed at its stacking position
      (position-above ?p-top ?p-bottom)  ; Top position must be directly above bottom position
    )
    :effect (and
      (on ?x ?y)
      (clear ?x)
      (not (clear ?y))
      (handempty)
      (not (holding ?x))
      (at-position ?x ?p-top)       ; Top block is now at the top position
      (not (position-free ?p-top))
    )
  )
)
