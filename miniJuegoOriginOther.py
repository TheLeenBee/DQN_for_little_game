# Game where things appear
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import time


def get_screen(positions, linewidth, screensize):
    # Given the middle positions of the items return a matrix that is a
    # screen. The positions have the following format:
    # [x, y, type, creationTime, freeVar]
    screen = np.zeros((screensize, screensize))

    for item in positions:
        x = item[0]
        y = item[1]
        w = int(np.ceil(linewidth/2))
        # x symbol
        if item[2] == 2:  # 1
            '''
            # Big cross
            screen[x-w:x+w, y-w:y+w] = 1
            screen[x+w:x+3*w, y+w:y+3*w] = 1
            screen[x+w:x+3*w, y-3*w:y-w] = 1
            screen[x-3*w:x-w, y-3*w:y-w] = 1
            screen[x-3*w:x-w, y+w:y+3*w] = 1


            # Small cross
            half = 5
            width = 4 * 2 * w
            length = 3 * 2 * w
            small = np.zeros((length+1, width+1))
            diag = np.arange(w, length + w)
            for i in range(2*w):
                small[np.arange(length), diag - w + i] = 1
                small[np.arange(length - 1, -1, -1), diag - w + i] = 1
            #print(small.shape, screen.shape, screen[x - half * w:x + half * w, y - half * w:y + half * w].shape, x, y)
            screen[x - 3 * w:x + 3 * w +1, y - 4 * w:y + 4 * w +1] = small
            '''


            large = 6
            smaller = 4
            # Small cross
            half = 5
            width = large * 2 * w
            length = smaller * 2 * w
            small = np.zeros((length+1, width+1))
            diag = np.arange(w, length + w)
            for i in range(4*w):
                small[np.arange(length), diag - w + i] = 1
                small[np.arange(length - 1, -1, -1), diag - w + i] = 1
            #print(small.shape, screen.shape, screen[x - half * w:x + half * w, y - half * w:y + half * w].shape, x, y)
            screen2 = np.zeros((screensize, screensize))
            screen2[x - smaller * w:x + smaller * w +1, y - large * w:y + large * w +1] = small
            screen += screen2


        # + symbol
        elif item[2] == 0:  # 2
            w = 2
            value = 1
            screen[x-w:x+w, y-w:y+w] = value
            screen[x+w:x+3*w, y-w:y+w] = value
            screen[x-w:x+w, y-3*w:y-w] = value
            screen[x-3*w:x-w, y-w:y+w] = value
            screen[x-w:x+w, y+w:y+3*w] = value

        # Player as a circle
        elif item[2] == 1:  # 0
            '''
            # screen[x-2*w:x+2*w, y-2*w:y+2*w] = 1
            # New player item
            value = 1
            screen[x - 2*w:x + 2*w, y - 2*w:y + 2*w] = value

            # circle
            width = 10
            xx, yy = np.mgrid[:width, :width]
            circle = (xx - 3 * w) ** 2 + (yy - 3 * w) ** 2
            #donut = np.logical_and(circle < (width/2 + width/2), circle > (width/2 - width/2))
            donut = np.logical_and(circle < (4 + 4), circle > (4 - 4))
            donut = donut[0:width/2 +1, 0:width/2 + 1]
            # write output
            #screen[x - width/2 * w:x + width/2 * w, y - width/2 * w:y + width/2 * w] = donut
            screen[x - (width / 4)-1:x + width / 4 +1, y - width /4 -1 :y  + width / 4 + 1 ] = donut
            #screen[x, y] = 0.5
            '''

            w = 3
            width = 46
            first = 3
            second = 2
            lower = 6
            higher = 30
            xx, yy = np.mgrid[:width, :width]
            #print(xx)
            circle = (xx - first * w) ** 2 + (yy - first * w) ** 2
            #print(circle)
            #donut = np.logical_and(circle < (width/2 + width/2), circle > (width/2 - width/2))
            donut = np.logical_and(circle < (higher), circle > (lower))
            #print(donut)
            donut = donut[0:round(width/3) +1, 0:round(width/3) + 1]
            #print(donut)
            # write output
            #screen[x - width/2 * w:x + width/2 * w, y - width/2 * w:y + width/2 * w] = donut
            screen2 = np.zeros((screensize, screensize))
            screen2[x - round(width / 6)-1:x + round(width / 6) +1, y - round(width /6) -1 :y  + round(width / 6) + 1 ] = donut
            screen += screen2
            screen[screen>1] = 1


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


def check_overlapping_player(positions, newpoint, linewidth, overlapfirst, stepsize):
    # Check if the new point would overlap with some existing point
    overlap = 0
    xs = positions[1:, 0]
    ys = positions[1:, 1]
    x = newpoint[0]
    y = newpoint[1]
    w = linewidth/2
    space = 9
    deleteditem = np.array((0, 0, 0, 0, 0))

    distx = abs(xs - x)
    disty = abs(ys - y)

    diff = np.sqrt((xs - x)**2 + (ys - y)**2)

    #print('distx', distx)
    #print('disty', disty)
    #print('w', w,'total dist',space*w)
    #raw_input('dist')
    '''
    for samp in range(len(distx)-1):
        if diff[samp] < space*w:
            #print(samp)
            #if distx[samp] < space*w and disty[samp] < space*w:
            deleteditem = positions[samp+1, :]
            positions = np.delete(positions, (samp + 1), axis=0)
            overlap = 1
            #print('deleted')
    '''

    addition = 0
    for samp in range(len(distx)):
        if diff[samp] < space*w:
            #print(samp)
            #if distx[samp] < space*w and disty[samp] < space*w:
            deleteditem = positions[samp+1-addition, :]
            positions = np.delete(positions, (samp + 1 - addition), axis=0)
            overlap = 1
	    break
            #print('deleted')

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


def initialise_item(screensize, positions, linewidth, stepsize, rounds, itemtype, timestep, amount):
    # Initialise items by giving them a random position
    nonew = 0
    middle = round(screensize/2)

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


def initialise_certain_item(x, y, screensize, positions, linewidth, stepsize, rounds, itemtype, timestep, amount):
    # Initialise items by giving them a random position
    nonew = 0
    middle = round(screensize/2)

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


def initialise_controlled_items(option, screensize, linewidth, stepsize, twoTypes):
    # Initialise some items in a controlled environment
    perRow = 5
    positions = np.zeros((1, 5))
    middle = round(screensize / 2)  # this is the center of the screen to start the player there
    # Player item
    positions[0, :] = np.array((middle, middle, 0, 0, 0))

    counter = 0
    if option == 1:
        # This is the first environement.
        # Create type one first
        for i in range(perRow):
            for j in range(perRow):
                if not (i == 2 and j == 2):
                    if not (i==3 and j == 2):
                        x = (j+1) * 14
                        y = (i+1) * 14
                        #positions = initialise_certain_item(x, y, screensize, positions, linewidth, stepsize, rounds=1, itemtype=j%2 +1, timestep=0,
                         #           amount=1)
                        if twoTypes:
                            positions = initialise_certain_item(x, y, screensize, positions, linewidth, stepsize,
                                                                rounds=1, itemtype=i%2+1, timestep=0, amount=1)
                        else:
                            positions = initialise_certain_item(x, y, screensize, positions, linewidth, stepsize,
                                                            rounds=1, itemtype= 1, timestep=0, amount=1)
                    #else:
                    #    positions[counter+1, :] = np.array((middle, middle, 0, 0, 0))

                    #counter += 1

    #itemtype= i%2+1

    return positions


def initialise_controlled_motion_items(screensize, linewidth, stepsize):
    # Initialise some items in a controlled environment
    perRow = 5
    positions = np.zeros((1, 5))
    middle = round(screensize / 2)  # this is the center of the screen to start the player there
    # Player item
    positions[0, :] = np.array((middle, 16, 0, 0, 0))

    for i in range(5):
        x = 14 + i*14
        y = 14 + i * 14
        positions = initialise_certain_item(x, y, screensize, positions, linewidth, stepsize,
                                                            rounds=1, itemtype= 1, timestep=0, amount=1)

    return positions


def initialise_items(numberItemsEach, screensize, linewidth, stepsize):
    # Initialise some items
    positions = np.zeros((1, 5))
    middle = round(screensize/2)  # this is the center of the screen to start the player there

    # Player item
    # positions[0, :] = np.array((random.randint(5, screensize-5), random.randint(5, screensize-5), 0, 0, 0))
    positions[0, :] = np.array((middle, middle, 0, 0, 0))

    # Initialise item of type 1
    positions = initialise_item(screensize, positions, linewidth, stepsize, rounds=1,  itemtype=2, timestep=0, amount=numberItemsEach[0])
    #positions = initialise_item(screensize, positions, linewidth, stepsize, rounds=2, itemtype=2, timestep=0, amount=numberItemsEach[0])
    #positions = initialise_item(screensize, positions, linewidth, stepsize, rounds=3, itemtype=2, timestep=0, amount=numberItemsEach[0])

    # Initialise item of type 2
    positions = initialise_item(screensize, positions, linewidth, stepsize, rounds=1,  itemtype=1, timestep=0, amount=numberItemsEach[1])
    #positions = initialise_item(screensize, positions, linewidth, stepsize, rounds=2,  itemtype=1, timestep=0, amount=numberItemsEach[1])
    #positions = initialise_item(screensize, positions, linewidth, stepsize, rounds=3,  itemtype=1, timestep=0, amount=numberItemsEach[1])

    return positions


def update_player(score, positions, linewidth, stepsize, screensize, goal, turn, direction, overlapfirst, controlled=0):
#def update_player(score, positions, linewidth, stepsize, screensize, goal, turn, direction, overlap, controlled=0):
    # Update the position of the player, this can be controlled
    # or at random as indicated by the variable controlled
    # Motions are labelled as:   |3|
    #                         |2|  |0|
    #                           |1|
    motionTable = np.array(((0, 1, 0, -1), (1, 0, -1, 0)))

    # if abs(positions[0, 0] - goal[0]) < stepsize * 2 and abs(goal[1] - positions[0, 1]) < stepsize * 2 :
    #    goal = [random.randint(linewidth, screensize-linewidth), random.randint(linewidth, screensize-linewidth)]

    if controlled:
        dx = motionTable[0, direction]
        dy = motionTable[1, direction]
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
        #print('still within boundaries')
        positions[0,:] = newpoint
    else:
        # print('out of boundaries. old direction: ', direction)
        # print('old direction: ', direction)
        direction = (direction + 2) % 4
        # print('new direction: ', direction)

    overlap, positions, deleteditem = check_overlapping_player(positions, newpoint, linewidth, overlapfirst, stepsize)
    if overlap:
        #print('score')
        if deleteditem[2] == 1:
            score = -10
            # score = 10
        else:
            score = 10 #10
    else:
        score = 0


    return positions, score, deleteditem, goal, turn, overlapfirst, direction


def plot_screen(frame, myobj, ax, timeStep):
    # Visualize the game
    plt.ion()
    if timeStep == 0:
        # Check if this is the first frame
        fig = plt.figure()
        ax = fig.add_subplot(111)
        myobj = plt.imshow(frame,interpolation='nearest', cmap='Greys')
        plt.ion()
        timeStep = 1
    else:
        # For all other time steps
        ax.clear()
        myobj = plt.imshow(frame,interpolation='nearest', cmap='Greys')
        plt.draw()

    return myobj, ax, timeStep


def update_items(positions, screensize, linewidth, stepsize, deleteditem, timestep):
    # As items disappear new ones appear. This can be changed later on
    # to items that have specific lifetimes
    motionTable = np.array(((0, 1, 0, -1), (1, 0, -1, 0)))

    # if abs(positions[0, 0] - goal[0]) < stepsize * 2 and abs(goal[1] - positions[0, 1]) < stepsize * 2 :
    #    goal = [random.randint(linewidth, screensize-linewidth), random.randint(linewidth, screensize-linewidth)]
    counter = 1
    for object in positions[1:,:]:
        direction = object[3]
        dx = motionTable[0, direction]
        dy = motionTable[1, direction]
        x = object[0]
        y = object[1]
        newx = x + dx*stepsize
        newy = y + dy*stepsize
        newpoint = np.array((newx, newy, 0, 0, 0))
        if not check_within_boundaries(newpoint, linewidth, stepsize, screensize):
            newpoint = np.array((newx, newy, object[2], direction, 0))
            positions[counter,:] = newpoint
        else:
            # print('out of boundaries. old direction: ', direction)
            # print('old direction: ', direction)
            direction = (direction + 2) % 4
            newpoint = np.array((newx, newy, object[2], direction, 0))
            positions[counter,:] = newpoint
            # print('new direction: ', direction)
        counter += 1


    ## Get details about deleted item
    if deleteditem[2] != 0:
        itemtype = deleteditem[2]
        newpoint = np.array((deleteditem[0], 60, itemtype, 0, 0))
        positions = np.vstack((positions, newpoint))
        #positions = initialise_item(screensize, positions, linewidth, stepsize, 1, itemtype, timestep, amount=1)

    return positions


def playgame(numberItems):
    # This function plays the game, for now the player uses random moves and
    # the game terminates after a fixed number of time steps.
    numberItemsEach = np.array((0, numberItems))
    screensize = 84
    stepsize = 2
    linewidth = 2
    timesteps = 300
    score = 0
    twoTypes=1

    # Calculate the positions
    #positions = initialise_items(numberItemsEach, screensize, linewidth, stepsize)
    #positions = initialise_controlled_items(1, screensize, linewidth, stepsize, twoTypes)
    positions = initialise_controlled_motion_items(screensize, linewidth, stepsize)
    screen = get_screen(positions, linewidth, screensize)

    # Initialise plotting
    firstFrame = np.zeros((screensize, screensize))
    myobj, ax, timeStep = plot_screen(firstFrame, 0, 0, 0)
    goal = 0
    turn = 0
    direction = random.randint(0,3)
    overlap = 0
    direction = 3

    for t in range(timesteps):
        print('time step: '+str(t))
        #direction = random.randint(0, 3)
        # Update the player. This also deletes item that it encounters
        # positions, score, deleteditem, goal, turn = update_player(score, positions, linewidth, stepsize, screensize, goal, turn, direction, controlled=0)
        positions, score, deleteditem, goal, turn, overlap, direction = update_player(score, positions, linewidth, stepsize,
                                                                           screensize, goal, turn, direction, overlap,
                                                                           controlled=1)
        positions = update_items(positions, screensize, linewidth, stepsize, deleteditem, t)
        #raw_input("Press Enter to continue...")
        #print('postions', positions)

        if abs(score) > 0:
            print(score)
        # Check if any item was deleted, if so replace it
        #if sum(deleteditem)>0:
         #   positions = update_items(positions,screensize,linewidth,stepsize,deleteditem,t)

        # Generate current screen and plot it
        screen = get_screen(positions, linewidth, screensize)
        myobj, ax, timeStep = plot_screen(screen, myobj, ax, timeStep)
        #plt.draw()
        #raw_input()


def get_frames(numberItemsEach, screensize, linewidth):
    # Function to get single frames from this game
    stepsize = 2
    # linewidth = 6

    # Calculate the positions
    positions = initialise_items(numberItemsEach, screensize, linewidth, stepsize)
    screen = get_screen(positions, linewidth, screensize)

    return screen, positions


def get_consecutive_framesArtificial(screensize, score, positions, linewidth, stepsize, t, numberItemsEach, direction, turn, overlap):
    # In case we want non consecutive frames
    # positions = initialise_items(numberItemsEach, screensize, linewidth, stepsize)
    # Function to get single frames from this game
    goal = 0
    positions, score, deleteditem, goal, turn, overlap, direction = update_player(score, positions, linewidth, stepsize,
                                                                       screensize, goal, turn, direction, overlap, controlled=1)

    # Check if any item was deleted, if so replace it
    if sum(deleteditem) > 0:
        positions = update_items(positions, screensize, linewidth, stepsize, deleteditem, t)

    # Get the actual screen to process
    screen = get_screen(positions, linewidth, screensize)
    convolutedScreen, coordsHighest = convert_convs(positions, screensize)

    # Update clock
    t += 1

    return screen, positions, score, t, convolutedScreen, coordsHighest, goal, turn, overlap, direction


def get_consecutive_frames(screensize, score, positions, linewidth, stepsize, t, goal, turn, direction, overlap):
    # Function to get single frames from this game
    positions, score, deleteditem, goal, turn, overlap, direction = update_player(score, positions, linewidth, stepsize, screensize, goal,
                                                              turn, direction, overlap, controlled=1)

    # Check if any item was deleted, if so replace it
    #if sum(deleteditem) > 0:
     #   positions = update_items(positions, screensize, linewidth, stepsize, deleteditem, t)

    # Get the actual screen to process
    screen = get_screen(positions, linewidth, screensize)

    # Update clock
    t += 1

    return screen, positions, score, t, goal, turn, overlap, direction


def get_consecutive_frames_motion(screensize, score, positions, linewidth, stepsize, t, goal, turn, direction, overlap):
    # Function to get single frames from this game
    positions, score, deleteditem, goal, turn, overlap, direction = update_player(score, positions, linewidth, stepsize, screensize, goal,
                                                              turn, direction, overlap, controlled=1)

    positions = update_items(positions, screensize, linewidth, stepsize, deleteditem, t)
    # Check if any item was deleted, if so replace it
    #if sum(deleteditem) > 0:
     #   positions = update_items(positions, screensize, linewidth, stepsize, deleteditem, t)

    # Get the actual screen to process
    screen = get_screen(positions, linewidth, screensize)

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
