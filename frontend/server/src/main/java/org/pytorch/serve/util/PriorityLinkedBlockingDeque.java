package org.pytorch.serve.util;

import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.Enumeration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PriorityLinkedBlockingDeque<T extends Prioritisable> {

    private static final Logger logger = LoggerFactory.getLogger(PriorityLinkedBlockingDeque.class);

    // lock and condition for waiting on empty queues
    final ReentrantLock lock = new ReentrantLock();
    private final Condition notEmpty = lock.newCondition();

    private final int queueSize;
    private final float highPrioProb;
    private final ConcurrentHashMap<Priority, LinkedBlockingDeque<T>> priorityDeques;

    public PriorityLinkedBlockingDeque(int queueSize, float highPrioProb) {

        this.queueSize = queueSize;
        this.highPrioProb = highPrioProb;
        this.priorityDeques = new ConcurrentHashMap<Priority, LinkedBlockingDeque<T>>();

        // initialize priority deques
        for (Priority priority : Priority.values()) { 
            this.priorityDeques.put(priority, new LinkedBlockingDeque<T>(queueSize));
        }
    }

    private LinkedBlockingDeque<T> getDequeForExtraction() {

        // always select deque max first if non-empty
        if (!this.priorityDeques.get(Priority.MAX).isEmpty()) {
            return this.priorityDeques.get(Priority.MAX);
        }

        boolean highNonEmpty = !this.priorityDeques.get(Priority.HIGH).isEmpty();

        // if both high and low are non-empty, make random selection
        if (highNonEmpty && !this.priorityDeques.get(Priority.LOW).isEmpty()) {
            if (ThreadLocalRandom.current().nextFloat() < this.highPrioProb) {
                return this.priorityDeques.get(Priority.HIGH);
            } else {
                return this.priorityDeques.get(Priority.LOW);
            }
        // if only high is non-empty, return high
        } else if (highNonEmpty) {
            return this.priorityDeques.get(Priority.HIGH);
        }

        // if both empty or only low non-empty, return low
        return this.priorityDeques.get(Priority.LOW);

    }

    private LinkedBlockingDeque<T> getDequeForInsertion(T p) {
        Priority priority = p.getPriority();
        LinkedBlockingDeque<T> dequeForInsertion = this.priorityDeques.get(priority);
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
        // return true iff all deques are empty
        return this.priorityDeques.reduceValues(Long.MAX_VALUE, LinkedBlockingDeque::isEmpty, Boolean::logicalAnd);
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

    /*
    this is exactly the same as the equivalent method in LinkedBlockingDeque, the difference is in the implementation of unlinkFirst()
    reference: https://github.com/openjdk/jdk17/blob/master/src/java.base/share/classes/java/util/concurrent/LinkedBlockingDeque.java
    */
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

    public String getQueueStatusString() {
        String response = "";
        for (Priority priority : Priority.values()) {
            int currentQueueStatus = this.priorityDeques.get(priority).size();
            response = response + priority.toString() + "=" + String.valueOf(currentQueueStatus) + ",";
        }
        response = response + "queueSize=" + String.valueOf(this.queueSize);
        return response;

    }
}
