import yaml
import requests
from pathlib import Path
from typing import Dict, List
from pyswip import Prolog

from coco.app.core.logger import get_logger

logger = get_logger(__name__)

RESOURCES_LIST = [
    {
        'hostname': 'ncsrd-worker3', 
        'available_cpu': 0.29999999999999993, 
        'available_memory': '10.81Gi'
    }, 
    {
        'hostname': 'ncsrd-worker', 
        'available_cpu': 2.2800000000000002, 
        'available_memory': '11.12Gi'
    }, 
    {
        'hostname': 'ubuntu', 
        'available_cpu': 1.83, 
        'available_memory': '10.70Gi'
    }, 
    {
        'hostname': 'duzunidis-vm1', 
        'available_cpu': 7.52, 
        'available_memory': '7.09Gi'
    }, 
    {
        'hostname': 'duzunidis-vm3', 
        'available_cpu': 6.72, 
        'available_memory': '3.00Gi'
    }, 
    {
        'hostname': 'duzunidis-vm2', 
        'available_cpu': 7.4399999999999995, 
        'available_memory': '5.19Gi'
    }, 
    {
        'hostname': 'cloud', 
        'available_cpu': 5.58, 
        'available_memory': '26.12Gi'
    }, 
    {
        'hostname': 'worker2', 
        'available_cpu': 3.68, 
        'available_memory': '12.63Gi'
    }, 
    {
        'hostname': 'worker1', 
        'available_cpu': 3.96, 
        'available_memory': '6.19Gi'
    }, 
    {
        'hostname': 'worker3', 
        'available_cpu': 3.84, 
        'available_memory': '5.84Gi'
    }
]

class Calibrator:
    """
    The Calibrator class is responsible for calibrating the given nLoTW scores using the available resources.
    """
    def __init__(self):
        """
        Initialize the Calibrator class.

        The Calibrator class takes no parameters and uses the CALIBRATION_DATA
        constant to determine the available resources for each node.

        :ivar calibration_data: A list of dictionaries, where each dictionary contains
            the hostname, available CPU and available memory for each node
        :type calibration_data: List[Dict[str, Any]]
        """
        pass
    
    def calibrate(self, nlotw_scores):
        """
        Calibrate the given nLoTW scores using the available resources.

        :param nlotw_scores: A dictionary where the keys are the labels and the values are lists of scores
        :type nlotw_scores: Dict[str, List[float]]

        :return: A list of tuples, where each tuple contains a label and its calibrated score
        :rtype: List[Tuple[str, float]]
        """
        # Add resources like this now
        resources = RESOURCES_LIST
        for resource in resources:
            logger.info(f"Hostname: {resource['hostname']}, CPU: {resource['available_cpu']:.2f}, Memory: {resource['available_memory']}")

         # Sort resources by available CPU and memory in descending order
        resources_sorted = sorted(
            resources,
            key=lambda r: (r['available_cpu'], float(r['available_memory'][:-2])),  # Convert memory to float for sorting
            reverse=True
        )

        # Select the infrastructure element with the most free resources
        best_resource = resources_sorted[0]
        available_cpu = best_resource['available_cpu']
        available_memory = best_resource['available_memory']

        logger.info(f"Using best resource: Hostname: {best_resource['hostname']}, CPU: {available_cpu:.2f}, Memory: {available_memory}")

        if available_memory.endswith('Gi'):
            available_memory = float(available_memory.replace('Gi', '')) * 1024
        elif available_memory.endswith('Mi'):
            available_memory = float(available_memory.replace('Mi', ''))

        calibrated_scores_list = []

        prolog = Prolog()
        path = Path(__file__).parent.parent / "core" / "general_flavor.pl"

        prolog = load_prolog_knowledge(path)

        for label, scores in nlotw_scores.items():
            avg_score = sum(scores) / len(scores)
            flavor = get_flavor(prolog, avg_score, available_cpu, available_memory)
            calibrated_score = get_flavor_upper_bound(prolog, flavor)

            # Store in the list
            calibrated_scores_list.append((label, calibrated_score))

            print(f"Score: {avg_score}, Calibrated Score: {calibrated_score}, Assigned Flavor: {flavor}")
            
        return calibrated_scores_list

def load_prolog_knowledge(base_file):
    """
    Loads a Prolog file and returns the associated Prolog instance.

    :param base_file: The path to the Prolog file
    :type base_file: str or Path
    :return: The Prolog instance
    :rtype: Prolog
    """
    prolog = Prolog()
    prolog.consult(base_file)
    return prolog

def get_flavor_upper_bound(prolog, flavor_name):
    """
    Calls the Prolog rule `flavor_upper_bound/2` to retrieve the max score of a given flavor.
    """
    query = f"flavor_upper_bound({flavor_name}, MaxScore)."
    result = list(prolog.query(query))

    if result:
        return result[0]["MaxScore"]  # Extract max score from Prolog response
    else:
        return None  # No matching flavor found
    
def get_flavor(prolog, score, cpu, memory):
    """
    Calls the Prolog rules `assign_flavor/4` and `fallback_flavor/4` to retrieve the flavor name that matches the given score, CPU, and memory.

    The rules are called in the following order:
    1. `assign_flavor/4` with the given score, CPU, and memory.
    2. If no matching flavor is found, `fallback_flavor/4` with the given CPU and memory.

    :param prolog: The Prolog instance
    :type prolog: Prolog
    :param score: The score to match
    :type score: float
    :param cpu: The CPU to match
    :type cpu: float
    :param memory: The memory to match
    :type memory: float or str
    :return: The flavor name that matches the given score, CPU, and memory, or "Insufficient Resources" if no matching flavor is found
    :rtype: str
    """
    query = list(prolog.query(f"assign_flavor({score}, {cpu}, {memory}, Flavor)"))
    if query:
        return query[0]["Flavor"]

    # Fallback logic
    fallback_query = list(prolog.query(f"fallback_flavor({cpu}, {memory}, Flavor)"))
    if fallback_query:
        return fallback_query[0]["Flavor"]

    return "Insufficient Resources"

if __name__ == "__main__":
    calibrator = Calibrator()
    calibrator.calibrate()

