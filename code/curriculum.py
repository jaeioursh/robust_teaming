def setCurriculumCoupling(data):
    schedule = data["Schedule"]
    episodeIndex = data["Episode Index"]
    generationSum = 0
    trainingCoupling = 1
    for coupling, duration in schedule:
        generationSum += duration
        trainingCoupling = coupling
        if generationSum > episodeIndex:
            break
    data["Test Coupling"] = data["Coupling"] 
    data["Coupling"] = trainingCoupling
            
def restoreCoupling(data):
    data["Coupling"] = data["Test Coupling"]
    
    
def setCurriculumWorldSize(data):
    schedule = data["Schedule"]
    episodeIndex = data["Episode Index"]
    generationSum = 0
    trainingCoupling = 1
    for worldSize, duration in schedule:
        generationSum += duration
        trainingWorldSize = worldSize
        if generationSum > episodeIndex:
            break
    data["Test World Width"] = data['World Width']
    data["Test World Length"] = data['World Length']
    data['World Width'] = trainingWorldSize
    data['World Length'] = trainingWorldSize
            
def restoreWorldSize(data):
    data['World Width'] = data["Test World Width"]
    data['World Length'] = data["Test World Length"]