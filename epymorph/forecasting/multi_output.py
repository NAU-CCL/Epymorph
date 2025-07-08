class MultiOutput:
    def __init__(
        self,
        rume,
        initial,
        visit_compartments,
        visit_events,
        home_compartments,
        home_events,
    ):
        self.rume = rume
        self.initial = initial
        self.visit_compartments = visit_compartments
        self.visit_events = visit_events
        self.home_compartments = home_compartments
        self.home_events = home_events
