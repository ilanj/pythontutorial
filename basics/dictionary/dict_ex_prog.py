import json
import pprint

student={
    "name":"bala",
    "id":36,
    43:"check",
    "address":"navalur"
}

print(student)
pprint.pprint(student)
student_asjson=json.dumps(student)
print(student_asjson)
for x,y in student.items():
    print(x,"\t",y)

