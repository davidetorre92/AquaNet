#!/bin/bash

echo 'EDA'
python -m bin.tasks.eda -c bin/config.ini

echo 'Core and periphery structure'
python -m bin.tasks.core_periphery_classification


