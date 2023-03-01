package org.pytorch.serve.util;

import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.function.BiFunction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PriorityLinkedBlockingDeque<T extends Prioritisable> {

    private static final Logger logger = LoggerFactory.getLogger(PriorityLinkedBlockingDeque.class);

    private int nPriorities;
    private ConcurrentHashMap<Integer, LinkedBlockingDeque<T>> priorityDeques;

    public PriorityLinkedBlockingDeque(int nPriorities, int queueSize) {

        this.nPriorities = nPriorities;

        priorityDeques = new ConcurrentHashMap<Integer, LinkedBlockingDeque<T>>();

        for (int i = 0; i < nPriorities; i++) {
            priorityDeques.put(i, new LinkedBlockingDeque<T>(queueSize));
        }
    }

    public PriorityLinkedBlockingDeque(int nPriorities) {
        this(nPriorities, Integer.MAX_VALUE);
    }

    public PriorityLinkedBlockingDeque() {
        this(1);
    }

    private LinkedBlockingDeque<T> getResponsibleDeque(T p) {
        int priority = p.getPriority();
        LinkedBlockingDeque<T> deque = priorityDeques.get(priority);

        if (deque == null) {
            logger.warn("Priority value not valid, setting to highest valid priority value.", priority);
            int newPriority = this.nPriorities - 1;
            p.setPriority(newPriority);
            deque = priorityDeques.get(newPriority);
        }

        return deque;
    }

    private LinkedBlockingDeque<T> getArbitraryDeque() {
        // TODO: Prioritised selection logic needed here
        return priorityDeques.get(0);
    }

    public boolean isEmpty() {
        Function<LinkedBlockingDeque<T>, Boolean> getIsEmpty = (LinkedBlockingDeque<T> deque) -> deque.isEmpty();
        BiFunction<Boolean, Boolean, Boolean> logicalAnd = (Boolean a, Boolean b) -> a && b;
        return priorityDeques.reduceValues(Long.MAX_VALUE, getIsEmpty, logicalAnd);
    }

    public boolean offer(T p) {
        return getResponsibleDeque(p).offer(p);
    }

    public void addFirst(T p) {
        getResponsibleDeque(p).addFirst(p);
    }

    public T poll(long timeout, TimeUnit unit) throws InterruptedException {
        return getArbitraryDeque().poll(timeout, unit);
    }

    public T poll() {
        return getArbitraryDeque().poll();
    }

}