# -*- coding: utf-8 -*-
class Error_Utils():
    __errors = dict([
        # Error genéricos
        (400, 'BAD REQUEST'),
        (403, 'Forbidden'),
        (500, 'INTERNAL SERVER ERROR')
    ])

    @staticmethod
    def get_message(id):
        if id in ErrorUtils.__errors:
            return ErrorUtils.__errors.get(id)
        else:
            return "UNEXPECTED ERROR. PLEASE CONTACT US AT {gilvan.praxedes@outlook.com}"