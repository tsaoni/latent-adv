from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

@dataclass
class Address:
    street: str
    city: str

# Merged data class
@dataclass
class PersonWithAddress:
    person: Person
    address: Address

person = Person(name="John", age=30)
address = Address(street="123 Main St", city="New York")
person_with_address = PersonWithAddress(person=person, address=address)

import pdb 
pdb.set_trace()