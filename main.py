import typing

import typeguard
import typing


# class Validator:
#     checker = typeguard.TypeChecker(config_type_check_lookup=True)
#
#     @classmethod
#     def assert_typing_deep(cls, value, expected_type) -> bool:
#         try:
#             Validator.checker.check_type(value, expected_type)
#             return True
#         except typeguard.TypeCheckError as a:
#             return False
#
#     @staticmethod
#     def assert_typing(value, expected_type) -> bool:
#         try:
#             typeguard.check_type(value, expected_type)
#             return True
#         except typeguard.TypeCheckError as a:
#             return False

# from typeguard import TypeChecker

def yy(x,y):
    print(f"{x} ____ {y}")
    return False

def main() -> None:
    # x = [1, 2, 3, "ala"]
    # print(Validator.assert_typing_deep(x, typing.List[int]))
    # print(Validator.assert_typing(x, typing.List[int]))
    x = [1, 2, 3, "ala"]
    # checker = TypeChecker(config_type_check_lookup=True)
    print(typeguard.check_type(x, typing.List[int], collection_check_strategy=typeguard.CollectionCheckStrategy.ALL_ITEMS, typecheck_fail_callback=yy))


if __name__ == '__main__':
    main()
