import pyclipper

subj = ((180, 200), (260, 200), (260, 150), (180, 150))

pco = pyclipper.PyclipperOffset()
pco.AddPath(subj, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

solution = pco.Execute(3)

print(solution)
