package org.pytorch.serve.util;

import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.function.BiFunction;
import java.util.Arrays;

import java.io.ObjectOutputStream;
import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PriorityLinkedBlockingDeque<T extends Prioritisable> {

    private static final Logger logger = LoggerFactory.getLogger(PriorityLinkedBlockingDeque.class);

    // Lock and condition for waiting on empty queues
    final ReentrantLock lock = new ReentrantLock();
    private final Condition notEmpty = lock.newCondition();

    private int nPriorities;
    private int sumPriorityWeights;
    private int[] weightedPriorityMap;
    private ConcurrentHashMap<Integer, LinkedBlockingDeque<T>> priorityDeques;

    // constructor
    public PriorityLinkedBlockingDeque(int nPriorities, int queueSize) {

        this.nPriorities = nPriorities;
        this.priorityDeques = new ConcurrentHashMap<Integer, LinkedBlockingDeque<T>>();

        // sum of 0 + 1 + 2 + ... + nPriorities - 1 via triangular sum
        this.sumPriorityWeights = ((this.nPriorities - 1) * this.nPriorities) / 2;
        this.weightedPriorityMap = new int[sumPriorityWeights];

        // initialize priority deques and weight map
        int keyStart = 0;
        for (int priority = 0; priority < this.nPriorities; priority++) {
            this.priorityDeques.put(priority, new LinkedBlockingDeque<T>(queueSize));
            if (priority > 0) {
                // priority weights are inverse priority values for now
                int priorityWeight = this.nPriorities - priority;
                for (int key = keyStart; key < keyStart + priorityWeight; key++) {
                    this.weightedPriorityMap[key] = priority;
                }
                keyStart += priorityWeight;
            }
        }
        logger.debug("PriorityLinkedBlockingDeque constructor finished");
    }

    private LinkedBlockingDeque<T> getDequeForExtraction() {

        logger.debug("getDequeForExtraction called");

        // always select deque 0 first if non-empty
        if (this.nPriorities == 1 || !this.priorityDeques.get(0).isEmpty()) {
            logger.debug("select 0");
            return this.priorityDeques.get(0);
        }

        // sample according to weight map
        int randInt = ThreadLocalRandom.current().nextInt(sumPriorityWeights);
        int randPriority = this.weightedPriorityMap[randInt];
        LinkedBlockingDeque<T> dequeForExtraction = this.priorityDeques.get(randPriority);

        logger.debug("sample random deque " + String.valueOf(randPriority));

        // it may happen that the sampled deque is empty, in that case proceed according to priority
        if (dequeForExtraction.isEmpty()) {
            logger.debug("random queue is empty");
            for (int priority = 1; priority < this.nPriorities; priority++) {
                LinkedBlockingDeque<T> priorityDeque = this.priorityDeques.get(priority);
                if (!priorityDeque.isEmpty()) {
                    logger.debug("found non-empty queue " + String.valueOf(priority));
                    return priorityDeque;
                }
            }
        }
        logger.debug("returning random queue");
        return dequeForExtraction;
    }

    private LinkedBlockingDeque<T> getDequeForInsertion(T p) {
        int priority = p.getPriority();
        LinkedBlockingDeque<T> dequeForInsertion = this.priorityDeques.get(priority);

        if (dequeForInsertion == null) {
            logger.warn("Priority value "  + String.valueOf(priority) + " not valid, setting to highest valid priority value " 
                + String.valueOf(this.nPriorities - 1) + ".");
            int newPriority = this.nPriorities - 1;
            p.setPriority(newPriority);
            dequeForInsertion = this.priorityDeques.get(newPriority);
        }

        return dequeForInsertion;
    }

    public boolean isEmpty() {
        logger.debug("isEmpty called");
        Function<LinkedBlockingDeque<T>, Boolean> getIsEmpty = (LinkedBlockingDeque<T> deque) -> deque.isEmpty();
        BiFunction<Boolean, Boolean, Boolean> logicalAnd = (Boolean a, Boolean b) -> a && b;
        return this.priorityDeques.reduceValues(Long.MAX_VALUE, getIsEmpty, logicalAnd);
    }

    public boolean offer(T p) {
        logger.debug("offer called");
        final ReentrantLock lock = this.lock;
        lock.lock();
        try {
            boolean itemInserted = getDequeForInsertion(p).offer(p);
            if (itemInserted) {
                // awaken one thread that is waiting for notEmpty condition
                notEmpty.signal();
            }
            return itemInserted;
        } finally {
            lock.unlock();
        }
    }

    public void addFirst(T p) {
        logger.debug("addFirst called");
        final ReentrantLock lock = this.lock;
        lock.lock();
        try {
            getDequeForInsertion(p).addFirst(p);
            // awaken one thread that is waiting for notEmpty condition
            notEmpty.signal();
        } finally {
            lock.unlock();
        }
    }

    // ideally, we would want to forward this to getDequeForExtraction().unlinkFirst(), but it is private
    // pollFirst() is a public method that forwards unlinkFirst(), so it's the next best alternative
    // reference: https://github.com/openjdk/jdk17/blob/master/src/java.base/share/classes/java/util/concurrent/LinkedBlockingDeque.java 
    private T unlinkFirst() {
        logger.debug("unlinkFirst called");
        return getDequeForExtraction().pollFirst();
    }

    // reference: https://github.com/openjdk/jdk17/blob/master/src/java.base/share/classes/java/util/concurrent/LinkedBlockingDeque.java
    public T poll(long timeout, TimeUnit unit) throws InterruptedException {
        logger.debug("poll called");
        long nanos = unit.toNanos(timeout);
        final ReentrantLock lock = this.lock;
        lock.lockInterruptibly();
        try {
            T x;
            while ( (x = unlinkFirst()) == null) {
                logger.debug("poll loop iter");
                if (nanos <= 0L) {
                    logger.debug("poll timed out");
                    return null;
                }
                nanos = notEmpty.awaitNanos(nanos);
            }
            logger.debug("poll return match");
            return x;
        } finally {
            lock.unlock();
        }
    }
}
