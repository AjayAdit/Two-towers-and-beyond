(define (problem two_blocks_move)
  (:domain blocksworld)
  (:objects r g - block)

  (:init
        (ontable r)
        (ontable g)
        (clear r)
        (clear g)
        (handempty)
  )

  (:goal (and (moved r) (moved g)))
)