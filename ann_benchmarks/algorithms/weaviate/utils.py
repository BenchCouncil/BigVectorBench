"""
Utility functions for weaviate
"""
import re


class Filter:
    """
    Filter class to create a filter object for weaviate query
    """
    def __init__(self, property_name):
        self.property_name = property_name
        self.conditions = []

    @staticmethod
    def by_property(property_name):
        """
        Create a new filter object by property name
        """
        return Filter(property_name)

    def greater(self, value):
        """
        Add greater than condition to the filter
        """
        self.conditions.append(f"greater_than({value})")
        return self

    def greater_or_equal(self, value):
        """
        Add greater than or equal condition to the filter
        """
        self.conditions.append(f"greater_or_equal({value})")
        return self

    def less(self, value):
        """
        Add less than condition to the filter
        """
        self.conditions.append(f"less_than({value})")
        return self

    def less_or_equal(self, value):
        """
        Add less than or equal condition to the filter
        """
        self.conditions.append(f"less_or_equal({value})")
        return self

    def equal(self, value):
        """
        Add equal condition to the filter
        """
        self.conditions.append(f"equal({value})")
        return self

    def notequal(self, value):
        """
        Add not equal condition to the filter
        """
        self.conditions.append(f"not_equal({value})")
        return self

    def __and__(self, other):
        if isinstance(other, CompositeFilter):
            return CompositeFilter("and", self, *other.filters)
        return CompositeFilter("and", self, other)

    def __or__(self, other):
        if isinstance(other, CompositeFilter):
            return CompositeFilter("or", self, *other.filters)
        return CompositeFilter("or", self, other)

    def __str__(self):
        conditions_str = " & ".join(self.conditions)
        return f"Filter.by_property('{self.property_name}').{conditions_str}"


class CompositeFilter:
    """
    Composite filter class to create a composite filter object for weaviate query
    """
    def __init__(self, operator, *filters):
        self.operator = operator
        self.filters = filters

    def __and__(self, other):
        if self.operator == "and":
            return CompositeFilter("and", *self.filters, other)
        return CompositeFilter("and", self, other)

    def __or__(self, other):
        if self.operator == "or":
            return CompositeFilter("or", *self.filters, other)
        return CompositeFilter("or", self, other)

    def __str__(self):
        op_symbol = '&' if self.operator == 'and' else '|'
        return f" {op_symbol} ".join(str(f) for f in self.filters)

def parse_condition(condition):
    """
    Parse a condition string and return a Filter object
    """
    pattern = re.compile(r"(\w+)\s*(==|>=|<=|>|<|!=)\s*(-?\d+)")
    match = pattern.match(condition)
    if not match:
        raise ValueError(f"Invalid condition: {condition}")

    property_name, operator, value = match.groups()
    value = int(value)

    if operator == ">=":
        return Filter.by_property(property_name).greater_or_equal(value)
    elif operator == "<=":
        return Filter.by_property(property_name).less_or_equal(value)
    elif operator == ">":
        return Filter.by_property(property_name).greater(value)
    elif operator == "<":
        return Filter.by_property(property_name).less(value)
    elif operator == "==":
        return Filter.by_property(property_name).equal(value)
    elif operator == "!=":
        return Filter.by_property(property_name).notequal(value)
    else:
        raise ValueError(f"Unsupported operator: {operator}")


def convert_conditions_to_filters(conditions) -> str:
    """
    Convert a condition string to a Filter object

    Args:
        conditions (str): condition string. Example: 
            - "age > 20 and height < 180 or weight == 70"
            - "text_length >= 3 and text_length <= 63 and unixtime >= 1485302400 and unixtime <= 1487894400 and star >= 4 and star <= 5"
    
    Returns:
        Filters (str): Filter objects for weaviate query
    """
    tokens = conditions.split()
    filters_stack = []
    operators_stack = []

    def apply_operator():
        operator = operators_stack.pop()
        right = filters_stack.pop()
        left = filters_stack.pop()
        if operator == "and":
            filters_stack.append(left & right)
        elif operator == "or":
            filters_stack.append(left | right)

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ["and", "or"]:
            while operators_stack and operators_stack[-1] == "and" and token == "or":
                apply_operator()
            operators_stack.append(token)
        else:
            # Reconstruct the condition from tokens
            condition = token
            i += 1
            while i < len(tokens) and tokens[i] not in ["and", "or"]:
                condition += " " + tokens[i]
                i += 1
            filters_stack.append(parse_condition(condition))
            continue
        i += 1

    while operators_stack:
        apply_operator()

    return str(filters_stack[0])
