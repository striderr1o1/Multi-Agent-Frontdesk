## Plainning Change in thread ID Design:

### Problem:
One link record has only one thread, which all the clients of that business share. 

### Solution:

Can use:
```python
thread_id = customer_unique_id + ":" + link_id
```

drop the column in links table "thread_id" since it is of no use.
