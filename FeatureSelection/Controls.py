class Control:
    def __init__(self, name, var_names):
        self.name = name
        self.var_names = var_names

    def join(self, other):
        return Control(name=f"{self.name} + {other.name}", var_names=self.var_names + other.var_names)


class NoControl(Control):
    def __init__(self):
        super().__init__(name='no_control', var_names=[])


class IncomeControl(Control):
    def __init__(self):
        super().__init__(name='income', var_names=['DEC_MED19'])


class EducationControl(Control):
    def __init__(self):
        super().__init__(name='education', var_names=['P19_ACT_DIPLMIN', 'P19_ACT_SUP2', 'P19_ACT_SUP5'])


class AgeControl(Control):
    def __init__(self):
        super().__init__(name='age',
                         var_names=['P19_POP1529', 'P19_POP3044', 'P19_POP4559', 'P19_POP6074', 'P19_POP75P'])
