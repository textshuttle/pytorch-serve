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

    // lock and condition for waiting on empty queues
    final ReentrantLock lock = new ReentrantLock();
    private final Condition notEmpty = lock.newCondition();

    private int nPriorities;
    private int sumPriorityWeights;
    private int[] weightedPriorityMap;
    private ConcurrentHashMap<Integer, LinkedBlockingDeque<T>> priorityDeques;

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
    }

    private LinkedBlockingDeque<T> getDequeForExtraction() {

        // always select deque 0 first if non-empty
        if (this.nPriorities == 1 || !this.priorityDeques.get(0).isEmpty()) {
            return this.priorityDeques.get(0);
        }

        // sample according to weight map
        int randInt = ThreadLocalRandom.current().nextInt(sumPriorityWeights);
        int randPriority = this.weightedPriorityMap[randInt];
        LinkedBlockingDeque<T> dequeForExtraction = this.priorityDeques.get(randPriority);

        // if sampled deque is empty, scan deques according to priority
        if (dequeForExtraction.isEmpty()) {
            for (int priority = 1; priority < this.nPriorities; priority++) {
                LinkedBlockingDeque<T> priorityDeque = this.priorityDeques.get(priority);
                if (!priorityDeque.isEmpty()) {
                    return priorityDeque;
                }
            }
        }
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

    /* 
    ideally, we would want to forward this to getDequeForExtraction().unlinkFirst(), but it is private
    pollFirst() is a public method that forwards unlinkFirst(), so it's the next best alternative
    reference: https://github.com/openjdk/jdk17/blob/master/src/java.base/share/classes/java/util/concurrent/LinkedBlockingDeque.java
    */
    private T unlinkFirst() {
        return getDequeForExtraction().pollFirst();
    }

    public boolean isEmpty() {
        Function<LinkedBlockingDeque<T>, Boolean> getIsEmpty = (LinkedBlockingDeque<T> deque) -> deque.isEmpty();
        BiFunction<Boolean, Boolean, Boolean> logicalAnd = (Boolean a, Boolean b) -> a && b;
        // return true iff all deques are empty
        return this.priorityDeques.reduceValues(Long.MAX_VALUE, getIsEmpty, logicalAnd);
    }

    public boolean offer(T p) {
        final ReentrantLock lock = this.lock;
        lock.lock();
        try {
            boolean itemInserted = getDequeForInsertion(p).offer(p);
            if (itemInserted) {
                // awaken one worker that is waiting for notEmpty condition
                notEmpty.signal();
            }
            return itemInserted;
        } finally {
            lock.unlock();
        }
    }

    public void addFirst(T p) {
        final ReentrantLock lock = this.lock;
        lock.lock();
        try {
            getDequeForInsertion(p).addFirst(p);
            // awaken one worker that is waiting for notEmpty condition
            notEmpty.signal();
        } finally {
            lock.unlock();
        }
    }

    public T poll(long timeout, TimeUnit unit) throws InterruptedException {
        long nanos = unit.toNanos(timeout);
        final ReentrantLock lock = this.lock;
        lock.lockInterruptibly();
        try {
            T x;
            while ( (x = unlinkFirst()) == null) {
                if (nanos <= 0L) {
                    return null;
                }
                // waits until notEmpty condition is signalled
                nanos = notEmpty.awaitNanos(nanos);
            }
            return x;
        } finally {
            lock.unlock();
        }
    }
}
