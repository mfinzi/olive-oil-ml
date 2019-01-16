from oil.tuning.slurmExecutor import SlurmPoolExecutor
import subprocess
import concurrent

# "Worker" functions.
def square(n):
    return n * n
def hostinfo():
    return subprocess.check_output('uname -a', shell=True)
def gpustat():
    return subprocess.check_output('gpustat', shell=True)
def cpu_count():
    import multiprocessing
    return multiprocessing.cpu_count()
def example_1():
    """Square some numbers on remote hosts!
    """
    with SlurmPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(square, n) for n in range(15)]
        for future in concurrent.futures.as_completed(futures):
            print((future.result()))

def example_2():
    """Get host identifying information about the servers running
    our jobs.
    """
    with SlurmPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(gpustat) for n in range(5)]
        print('Some cluster nodes:')
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

# def example_3():
#     """Demonstrates the use of the map() convenience function.
#     """
#     exc = SlurmExecutor(False)
#     print((list(cfut.map(exc, square, [5, 7, 11]))))

if __name__ == '__main__':
    #example_1()
    example_2()
    #example_3()
