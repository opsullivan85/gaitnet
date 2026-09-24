# Start a swing of one leg now.

uint8 FL=0
uint8 FR=1
uint8 RL=2
uint8 RR=3

uint8 leg
float32[3] target       # m, the foothold relative to the leg's hip, in the hip's yaw frame at
                        # the answered observation's stamp; z is the terrain surface there (the
                        # sole touches it: add the foot's radius for the foot's centre). A spot
                        # on the ground: pin it in odometry from the pose at that stamp and
                        # swing to it, don't re-apply it to the hip later
float32 duration        # s, from lift-off to touchdown
