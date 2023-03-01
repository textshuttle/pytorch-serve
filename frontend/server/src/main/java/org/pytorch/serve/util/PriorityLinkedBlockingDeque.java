package org.pytorch.serve.util;

import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
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
        // TODO: I think the proper way of doing this would be ConcurrentHashMap.reduceValues
        return priorityDeques.get(0).isEmpty();
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