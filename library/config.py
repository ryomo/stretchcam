import os
from configparser import ConfigParser


class Config(ConfigParser):
    """
    Configuration file reader and writer.
    * Read the default values from default.ini.
    * Read user.ini if exists and overwrite the default values.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        default_conf_path = "config/default.ini"
        self.user_conf_path = "config/user.ini"

        self.read(
            [
                default_conf_path,
                self.user_conf_path,
            ]
        )


# Test
if __name__ == "__main__":
    _dir = os.path.dirname(__file__)
    os.chdir(_dir + "/..")
    config = Config()
    print(config.get("app", "name"))
