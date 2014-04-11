import threading, Queue, subprocess

from collections import namedtuple, deque, Counter

import logging
import numpy as np
import itertools
import fluidsynth
import json
import random
import signal
import sys
import time

# Global Config
CONFIG = "config.json"
logger = logging.getLogger('vimusical')

InputEvent = namedtuple('InputEvent', ['input', 'time'])
SoundStream = namedtuple('SoundStream', ['key', 'vel', 'duration'])

class Player:
    '''Responsible for initializing fluidsynth and playing the soundstream.
    '''
    def __init__(self, sfConfig):
        self.playing = []
        self.fs = fluidsynth.Synth(gain=1.0)
        sfid = self.fs.sfload(sfConfig['file'])
        self.fs.program_select(chan=0, 
            sfid=sfid, 
            bank=0, 
            preset=sfConfig['presetId'])
        self.fs.start()

        # states
        self.playing = {}

    def stopSound(self, note):
        self.fs.noteoff(0, note)
        del self.playing[note]

    def play(self, soundStreams):

        for group in soundStreams[-1:3]:
            logger.debug('SoundStream: %s', group)
            keyCount = 0
            for note in group:
                if keyCount > 2: break

                if note.key not in self.playing:
                    self.fs.noteon(0, note.key, note.vel)
                    threading.Timer(note.duration/950, 
                        self.stopSound, (note.key,)).start()

                    keyCount += 1
                    self.playing[note.key] = True

class Keyboard:
    '''Responsible for deciphering key to soundstreams.'''
    def __init__(self):
        self.BEGIN_KEY = 36
        self.END_KEY = 96

        # number of keys to peek
        self.PEEK_WINDOW = 3
        self.DELTA_WINDOW = 500

        self.dynamicKeyMap = {}

    def _decipherKeys(self, group):
        keys, delta, duration = group

        soundKeys = []
        for key in keys:
            if key not in self.dynamicKeyMap:
                self.dynamicKeyMap[key] = random.randint(self.BEGIN_KEY, self.END_KEY)
            soundKeys.append(self.dynamicKeyMap[key])
        soundVel = random.randint(60, 127) if delta > 250 else random.randint(20, 70)

        return map(lambda args: SoundStream(*args),
                zip(soundKeys, [soundVel] * len(keys), [duration] * len(keys)))

    def interprete(self, inputEvents):
        self.key = 0
        def groupByKey(delta):
            if delta > self.DELTA_WINDOW:
                retKey = self.key
                self.key += 1
                return retKey
            return self.key

        # group by time delta
        inputEventsTime = map(lambda x: x.time, inputEvents)
        inputEventsChar = map(lambda x: x.input, inputEvents)
        timeDelta = list(np.diff(np.array(inputEventsTime), n=1)) + [0]
        group = itertools.groupby(timeDelta, key=groupByKey)

        logger.debug('Inputs: %s' % inputEventsChar)
        logger.debug('Delta: %s' % ['%.2f' % t for t in (timeDelta)])

        startInd = 0
        groupedInputEvents = []
        for k, g in itertools.groupby(timeDelta, key=groupByKey):
            groupLen = len(list(g))

            chars = "".join(inputEventsChar[startInd:(startInd + groupLen)])
            meanDelta = np.array(timeDelta[startInd:(startInd + groupLen)]).mean()
            duration = meanDelta

            groupedInputEvents.append((chars, meanDelta, duration))
            startInd += groupLen

        return map(lambda group: self._decipherKeys(group), groupedInputEvents)

class Stats:
    def __init__(self):
        pass

    @staticmethod
    def update(inputEvents):
        inputEvents = list(inputEvents)
        inputEventsTime = map(lambda x: x.time, inputEvents)
        speeds = np.diff(np.array(inputEventsTime), n=1)

        if len(speeds) > 3:
            accls = np.gradient(speeds)
        else:
            accls = np.array([])

        speed = speeds.mean()
        accl = accls.mean()

        logger.debug('Speed: %f' % speed)
        logger.debug('Accle: %f' % accl)

        return speed, accl

class Vimusical:
    def __init__(self, config):
        self.exitEvent = threading.Event()
        self.tailQ = Queue.Queue(maxsize=10)
        self.inputfile = config['inputfile']

        # Init fs
        sfConfig = config['soundfont']

        # Internal State
        self.inputEventBuffer = deque(maxlen=12)
        self.inputEventSpeed = deque(maxlen=12)
        self.inputEventAccl = deque(maxlen=12)

        self.counter = Counter()

        self.player = Player(config['soundfont'])
        self.keyboard = Keyboard()

    def __tailStream(self, fn):
        f = subprocess.Popen(["tail", "-F", fn],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while not self.exitEvent.is_set():
            inputKey = f.stdout.read(1)
            timestampe = time.time() * 1000
            self.tailQ.put(InputEvent(inputKey, timestampe))

    def __monitor(self):
        self.tailThread = threading.Thread(target=self.__tailStream,
            args=(self.inputfile,))
        self.tailThread.start()

    def __process(self, inputEvent):
        self.inputEventBuffer.append(inputEvent)
        avgSpeed, avgAccl = Stats.update(self.inputEventBuffer)

        self.inputEventSpeed.append(avgSpeed)
        self.inputEventAccl.append(avgAccl)
        self.counter.update(inputEvent.input)

        soundStreams = self.keyboard.interprete(list(self.inputEventBuffer))
        self.player.play(soundStreams)

    def __run(self):
        try:
            while True:
                inputEvent = self.tailQ.get()
                self.__process(inputEvent)
        except KeyboardInterrupt as e:
            self.exitEvent.set()
            sys.exit(0)

    def run(self):
        self.__monitor()
        self.__run()

if __name__ == "__main__":
    config = json.load(open(CONFIG))

    logging.basicConfig()
    logger.setLevel(config['loglevel'])

    vimusical = Vimusical(config)
    vimusical.run()
