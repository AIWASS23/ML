class Item:
  
    """Função construtora com preço e desconto"""
  
    def __init__(self, price, discount_strategy = None):
          
        """estratégia de preço e desconto"""
          
        self.price = price
        self.discount_strategy = discount_strategy
          
    """Uma função separada para preço após desconto"""
  
    def price_after_discount(self):
          
        if self.discount_strategy:
            discount = self.discount_strategy(self)
        else:
            discount = 0
              
        return self.price - discount
  
    def __repr__(self):
          
        statement = "Preço: {}, preço após desconto: {}"
        return statement.format(self.price, self.price_after_discount())
  
"""função dedicada ao desconto na venda"""
def on_sale_discount(order):
      
    return order.price * 0.25 + 20
  
"""função dedicada a 20% de desconto"""
def twenty_percent_discount(order):
      
    return order.price * 0.20
  
"""main"""
if __name__ == "__main__":
  
    print(Item(20000))
      
    """com estratégia de desconto de 20% de desconto"""
    print(Item(20000, discount_strategy = twenty_percent_discount))
  
    """com estratégia de desconto como desconto na venda"""
    print(Item(20000, discount_strategy = on_sale_discount))