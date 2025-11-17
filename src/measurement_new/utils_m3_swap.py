from openfermion import QubitOperator

def XandY_augment(H, Nqubits):
    Haugmented = QubitOperator()
    for term, coef in H.terms.items():
        Haugmented += coef * QubitOperator(term) * (
            QubitOperator(f'X{Nqubits}') + 1j * QubitOperator(f'Y{Nqubits}')
        )
    return Haugmented

def y_parity_of_term(term):
    num = 0
    for letter in term:
        if letter[1] == 'Y':
            num += 1
    return num % 2

def XorY_augment(H, Nqubits):
    Haugmented = QubitOperator()
    for term, coef in H.terms.items():
        par = y_parity_of_term(term)
        if par == 0:
            Haugmented += coef * QubitOperator(term) * QubitOperator(f'X{Nqubits}')
        else:
            Haugmented += coef * QubitOperator(term) * 1j * QubitOperator(f'Y{Nqubits}')
    return Haugmented
