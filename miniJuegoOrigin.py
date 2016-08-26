# Game where things appear
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import time


def get_screen(positions):
    # Given the middle positions of the items return a matrix that is a
    # screen. The positions have the following format:
    # [x, y, type, creationTime, freeVar]
    linewidth = 2
    screensize = 84
    screen = np.zeros((screensize, screensize))
    value = 255

    for item in positions:
        x = item[0]
        y = item[1]
        w = int(np.ceil(linewidth/2))
        # x symbol
        if item[2] == 2:  # 1
            large = 6
            smaller = 4
            # Small cross
            width = large * 2 * w
            length = smaller * 2 * w
            small = np.zeros((length+1, width+1))
            diag = np.arange(w, length + w)
            for i in range(4*w):
                small[np.arange(length), diag - w + i] = value
                small[np.arange(length - 1, -1, -1), diag - w + i] = value
            # print(small.shape, screen.shape, screen[x - half * w:x + half * w, y - half * w:y + half * w].shape, x, y)
            screen2 = np.zeros((screensize, screensize))
            screen2[x - smaller * w:x + smaller * w + 1, y - large * w:y + large * w + 1] = small
            screen += screen2

        # + symbol
        elif item[2] == 1:  # 2
            w = 2
            screen[x-w:x+w, y-w:y+w] = value
            screen[x+w:x+3*w, y-w:y+w] = value
            screen[x-w:x+w, y-3*w:y-w] = value
            screen[x-3*w:x-w, y-w:y+w] = value
            screen[x-w:x+w, y+w:y+3*w] = value

        # Player as a circle
        elif item[2] == 0:  # 0

            w = 3
            width = 46
            first = 3
            lower = 6
            higher = 30
            xx, yy = np.mgrid[:width, :width]
            circle = (xx - first * w) ** 2 + (yy - first * w) ** 2
            donut = np.logical_and(circle < higher, circle > lower)
            donut = donut[0:round(width/3) + 1, 0:round(width/3) + 1]
            # write output
            screen2 = np.zeros((screensize, screensize))
            screen2[x - round(width / 6)-1:x + round(width / 6) + 1, 
                    y - round(width / 6) - 1:y + round(width / 6) + 1] = donut * value
            screen += screen2
            screen[screen > value] = value
            
    return screen


def check_overlapping(positions, newpoint, linewidth):
    # Check if the new point would overlap with some existing point
    overlap = 0
    xs = positions[:, 0]
    ys = positions[:, 1]
    x = newpoint[0]
    y = newpoint[1]
    w = linewidth/2
    space = 13

    # Check if one of the items is the player, that is smaller
    if newpoint[2] == 0:
        space = 5

    distx = abs(xs - x)
    disty = abs(ys - y)

    for samp in range(len(distx)):
        if distx[samp] < space*w and disty[samp] < space*w:
            overlap = 1

    return overlap


def check_overlapping_player(positions, newpoint, linewidth):
    # Check if the new point would overlap with some existing point
    '''
    overlap = 0
    xs = positions[1:, 0]
    ys = positions[1:, 1]
    x = newpoint[0]
    y = newpoint[1]
    w = linewidth/2
    space = 9
    deleteditem = np.array((0, 0, 0, 0, 0))

    distx = abs(xs - x)

    diff = np.sqrt((xs - x)**2 + (ys - y)**2)

    for samp in range(len(distx)-1):
        if diff[samp] < space*w:
            deleteditem = positions[samp+1, :]
            positions = np.delete(positions, (samp + 1), axis=0)
            overlap = 1

    return overlap, positions, deleteditem'''
    overlap = 0
    xs = positions[1:, 0]
    ys = positions[1:, 1]
    x = newpoint[0]
    y = newpoint[1]
    w = linewidth / 2
    space = 9
    deleteditem = np.array((0, 0, 0, 0, 0))

    distx = abs(xs - x)
    disty = abs(ys - y)

    diff = np.sqrt((xs - x) ** 2 + (ys - y) ** 2)
    addition = 0
    #raw_input()
    for samp in range(len(distx)):
        if diff[samp] < space * w:
            deleteditem = positions[samp + 1 - addition, :]
            positions = np.delete(positions, (samp + 1 - addition), axis=0)
            overlap = 1
            raw_input()
            break

    return overlap, positions, deleteditem


def check_within_boundaries(newpoint, linewidth, stepsize, screensize):
    # Check if new point is within boundaries
    # First check what type of item it is
    if newpoint[2] == 0:  # player
        dist = 10
    else:
        dist = 6

    # Params
    x = newpoint[0]
    y = newpoint[1]
    w = linewidth/2
    s = screensize

    if dist*w + stepsize < x and x < s - (dist*w + stepsize) and dist*w + stepsize < y and y < s - (dist*w + stepsize):
        out = 0
    else:
        out = 1

    return out


def initialise_item(screensize, positions, linewidth, stepsize, itemtype, amount):
    # Initialise items by giving them a random position
    nonew = 0

    # Create each of the items
    for i in range(amount):
        x = random.randint(1, screensize)
        y = random.randint(1, screensize)

        '''
        # This arrangement creates parallel lines of items that are sorted by type:
        if itemtype == 2:
            y = 30 + 20*i
            x = middle - rounds * 20
        else:
            y = 30 + 20*i
            x = middle + rounds * 20
        '''

        newpoint = np.array((x, y, itemtype, 0, 0))
        counter = 0
        while check_overlapping(positions, newpoint, linewidth) or check_within_boundaries(newpoint, linewidth, 
                                                                                           stepsize, screensize):
            x = random.randint(1, screensize)
            y = random.randint(1, screensize)
            newpoint = np.array((x, y, itemtype, 0, 0))
            counter += 1
            if counter > 100:
                # If we struggle to find a point we won't add it
                nonew = 1
                break
        if not nonew:
            positions = np.vstack((positions, newpoint))

    return positions


def initialise_certain_item(x, y, screensize, positions, linewidth, stepsize, itemtype):
    # Initialise items by giving them a random position
    nonew = 0

    newpoint = np.array((x, y, itemtype, 0, 0))
    counter = 0
    while check_overlapping(positions, newpoint, linewidth) or check_within_boundaries(newpoint, linewidth, stepsize, screensize):
        x = random.randint(1, screensize)
        y = random.randint(1, screensize)
        newpoint = np.array((x, y, itemtype, 0, 0))
        counter += 1
        if counter > 100:
            # If we struggle to find a point we won't add it
            nonew = 1
            break
    if not nonew:
        positions = np.vstack((positions, newpoint))

    return positions


def initialise_controlled_items(two_types):
    # Parameters are set for the DQN
    option = 1
    screensize = 84
    linewidth = 2
    stepsize = 1
    #two_types = 1
    # Initialise some items in a controlled environment
    per_row = 5
    positions = np.zeros((1, 5))
    middle = round(screensize / 2)  # this is the center of the screen to start the player there
    # Player item
    positions[0, :] = np.array((middle, middle, 0, 0, 0))

    if option == 1:
        # This is the first environement.
        # Create type one first
        for i in range(per_row):
            for j in range(per_row):
                if not (i == 2 and j == 2):
                    if not (i == 3 and j == 2):
                        x = (j+1) * 14
                        y = (i+1) * 14
                        if two_types:
                            positions = initialise_certain_item(x, y, screensize, positions, linewidth, stepsize,
                                                                itemtype=i % 2+1)
                        else:
                            positions = initialise_certain_item(x, y, screensize, positions, linewidth, stepsize,
                                                                itemtype=1)

    return positions


def initialise_controlled_motion_items(screensize, linewidth, stepsize):
    # Initialise some items in a controlled environment
    positions = np.zeros((1, 5))
    middle = round(screensize / 2)  # this is the center of the screen to start the player there
    # Player item
    positions[0, :] = np.array((middle, 16, 0, 0, 0))

    for i in range(5):
        x = 14 + i*14
        y = 14 + i * 14
        positions = initialise_certain_item(x, y, screensize, positions, linewidth, stepsize, itemtype=1)

    return positions


def initialise_items(two_types):
    number_items_each = np.array((6, 6))
    screensize = 84
    linewidth = 2
    stepsize = 1
    # Initialise some items
    positions = np.zeros((1, 5))
    middle = round(screensize/2)  # this is the center of the screen to start the player there

    # Player item
    positions[0, :] = np.array((middle, middle, 0, 0, 0))

    # Initialise item of type 1
    positions = initialise_item(screensize, positions, linewidth, stepsize, itemtype=1, amount=number_items_each[0])

    if two_types:
        positions = initialise_item(screensize, positions, linewidth, stepsize, itemtype=2, amount=number_items_each[0])
    else:
        positions = initialise_item(screensize, positions, linewidth, stepsize, itemtype=1, amount=number_items_each[1])

    return positions


def update_player(positions, linewidth, stepsize, screensize, goal, turn, direction, overlapfirst, controlled=0):
    # Update the position of the player, this can be controlled
    # or at random as indicated by the variable controlled
    # Motions are labelled as:   |3|
    #                         |2|  |0|
    #                           |1|
    motion_table = np.array(((0, 1, 0, -1), (1, 0, -1, 0)))

    # if abs(positions[0, 0] - goal[0]) < stepsize * 2 and abs(goal[1] - positions[0, 1]) < stepsize * 2 :
    #    goal = [random.randint(linewidth, screensize-linewidth), random.randint(linewidth, screensize-linewidth)]

    if controlled:
        dx = motion_table[0, direction]
        dy = motion_table[1, direction]
    else:
        # If not controlled just move to the right across the screen
        dx = 0
        dy = 1

    # Check if this movement is possible otherwise ignore
    x = positions[0, 0]
    y = positions[0, 1]
    newx = x + dx*stepsize
    newy = y + dy*stepsize
    newpoint = np.array((newx, newy, 0, 0, 0))
    if not check_within_boundaries(newpoint, linewidth, stepsize, screensize):
        positions[0, :] = newpoint
    else:
        direction = (direction + 2) % 4

    overlap, positions, deleteditem = check_overlapping_player(positions, newpoint, linewidth)
    if overlap:
        if deleteditem[2] == 1:
            score = -10
        else:
            score = 10
    else:
        score = 0

    return positions, score, deleteditem, goal, turn, overlapfirst, direction


def plot_screen(frame, ax, time_step):
    # Visualize the game
    plt.ion()
    if time_step == 0:
        # Check if this is the first frame
        fig = plt.figure()
        ax = fig.add_subplot(111)
        myobj = plt.imshow(frame, interpolation='nearest', cmap='Greys')
        plt.ion()
        time_step = 1
    else:
        # For all other time steps
        ax.clear()
        myobj = plt.imshow(frame, interpolation='nearest', cmap='Greys')
        plt.draw()

    return myobj, ax, time_step


def update_items(positions, screensize, linewidth, stepsize, deleteditem):
    # As items disappear new ones appear. This can be changed later on
    # to items that have specific lifetimes
    motion_table = np.array(((0, 1, 0, -1), (1, 0, -1, 0)))

    counter = 1
    for object in positions[1:, :]:
        direction = object[3]
        dx = motion_table[0, direction]
        dy = motion_table[1, direction]
        x = object[0]
        y = object[1]
        newx = x + dx*stepsize
        newy = y + dy*stepsize
        newpoint = np.array((newx, newy, 0, 0, 0))
        if not check_within_boundaries(newpoint, linewidth, stepsize, screensize):
            newpoint = np.array((newx, newy, object[2], direction, 0))
            positions[counter, :] = newpoint
        else:
            direction = (direction + 2) % 4
            newpoint = np.array((newx, newy, object[2], direction, 0))
            positions[counter, :] = newpoint
        counter += 1

    # Get details about deleted item
    if deleteditem[2] != 0:
        itemtype = deleteditem[2]
        newpoint = np.array((deleteditem[0], 60, itemtype, 0, 0))
        positions = np.vstack((positions, newpoint))

    return positions


def playgame(number_items):
    # This function plays the game, for now the player uses random moves and
    # the game terminates after a fixed number of time steps.
    screensize = 84
    stepsize = 2
    linewidth = 2
    timesteps = 300

    # Calculate the positions
    positions = initialise_controlled_motion_items(screensize, linewidth, stepsize)

    # Initialise plotting
    first_frame = np.zeros((screensize, screensize))
    myobj, ax, time_step = plot_screen(first_frame, 0, 0)
    goal = 0
    turn = 0
    overlap = 0
    direction = 3

    for t in range(timesteps):
        print('time step: '+str(t))
        # Update the player. This also deletes item that it encounters
        positions, score, deleteditem, goal, turn, overlap, direction = update_player(positions, linewidth, stepsize,
                                                                                      screensize, goal, turn, direction,
                                                                                      overlap, controlled=1)
        positions = update_items(positions, screensize, linewidth, stepsize, deleteditem)

        if abs(score) > 0:
            print(score)
        # Check if any item was deleted, if so replace it
        # if sum(deleteditem)>0:
        #   positions = update_items(positions,screensize,linewidth,stepsize,deleteditem,t)

        # Generate current screen and plot it
        screen = get_screen(positions)
        myobj, ax, time_step = plot_screen(screen, ax, time_step)


def get_frames(number_items_each, screensize, linewidth):
    # Function to get single frames from this game
    stepsize = 2

    # Calculate the positions
    positions = initialise_items(number_items_each, screensize, linewidth, stepsize)
    screen = get_screen(positions)

    return screen, positions


def get_consecutive_framesArtificial(screensize, score, positions, linewidth, stepsize, t, numberItemsEach, direction, turn, overlap):
    # In case we want non consecutive frames
    # positions = initialise_items(numberItemsEach, screensize, linewidth, stepsize)
    # Function to get single frames from this game
    goal = 0
    positions, score, deleteditem, goal, turn, overlap, direction = update_player(positions, linewidth, stepsize,
                                                                       screensize, goal, turn, direction, overlap, controlled=1)

    # Check if any item was deleted, if so replace it
    if sum(deleteditem) > 0:
        positions = update_items(positions, screensize, linewidth, stepsize, deleteditem)

    # Get the actual screen to process
    screen = get_screen(positions)
    convolutedScreen, coordsHighest = convert_convs(positions, screensize)

    # Update clock
    t += 1

    return screen, positions, score, t, convolutedScreen, coordsHighest, goal, turn, overlap, direction


def get_consecutive_frames(direction, positions, t):
    screensize = 84
    score = 0
    linewidth = 2
    stepsize = 1
    goal = 0
    turn = 0
    overlap = 0
    # Function to get single frames from this game
    positions, score, deleteditem, goal, turn, overlap, direction = update_player(positions, linewidth, stepsize, screensize, goal,
                                                              turn, direction, overlap, controlled=1)

    # Check if any item was deleted, if so replace it
    if sum(deleteditem) > 0:
        positions = update_items(positions, screensize, linewidth, stepsize, deleteditem)

    # Get the actual screen to process
    screen = get_screen(positions)

    # Update clock
    t += 1

    return score, positions, t


def get_consecutive_frames_motion(screensize, score, positions, linewidth, stepsize, t, goal, turn, direction, overlap):
    # Function to get single frames from this game
    positions, score, deleteditem, goal, turn, overlap, direction = update_player(positions, linewidth, stepsize, screensize, goal,
                                                              turn, direction, overlap, controlled=1)

    positions = update_items(positions, screensize, linewidth, stepsize, deleteditem)
    # Check if any item was deleted, if so replace it
    #if sum(deleteditem) > 0:
     #   positions = update_items(positions, screensize, linewidth, stepsize, deleteditem, t)

    # Get the actual screen to process
    screen = get_screen(positions)

    # Update clock
    t += 1

    return screen, positions, score, t, goal, turn, overlap, direction


def convert_convs(positions, screensize):
    # Quick function to generate convolutions, since lasagne is not working at home.
    # Load the pixels for the different types of items.
    finalSize = 25
    f = open('peruba/peruba_c1-15_c2-5_f-10_p1-15_p2-51.pckl')
    l = pickle.load(f)
    im = l[2]
    coordsHighest = np.zeros((positions.shape[0], 2))

    cross = im[0, :, 7, 3]
    square = im[0, :, 5, 5 ]
    ex = im[0, : , 3, 5]

    convolutedScreen = np.zeros((1, 10, finalSize, finalSize))
    counter = 0

    for item in positions:
        x = int(np.floor(item[0]*finalSize/screensize))
        y = int(np.floor(item[1]*finalSize/screensize))
        if not (x == 0 and y == 0):
            coordsHighest[counter, 0] = x
            coordsHighest[counter, 1] = y
            counter += 1
        if item[2] == 1:
            convolutedScreen[0, :, x, y] += ex
        elif item[2] == 2:
            convolutedScreen[0, :, x, y] += cross
        else:
            convolutedScreen[0, :, x, y] += square

    return convolutedScreen, coordsHighest


if __name__ == '__main__':
    numberItems = 7
    playgame(20)
