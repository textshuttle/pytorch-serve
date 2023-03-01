package org.pytorch.serve.util;

public interface Prioritisable {

    public int getPriority();
    public void setPriority(int priority);

}