import pyhf

model = pyhf.Model(
    {
        "channels": [
            {
                "name": "singlechannel",
                "samples": [
                    {
                        "name": "signal",
                        "data": [5.0, 10.0],
                        "modifiers": [
                            {"name": "mynormfactor", "type": "normfactor", "data": None}
                        ],
                    }
                ],
            }
        ]
    },
    poi_name=None,
)

print(f"  aux data: {model.config.auxdata}")
print(f"   nominal: {model.expected_data([1.0])}")
print(f"2x nominal: {model.expected_data([2.0])}")
print(f"3x nominal: {model.expected_data([3.0])}")

